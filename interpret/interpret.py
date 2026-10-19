# interpret/interpret.py: interpretability rig; extracts hooked activations to chunks, then runs the configured methods on them (see README.md).
import os
import sys
import math
import json
import hashlib
import subprocess
import random
import yaml
import torch
from torchvision import transforms
from pathlib import Path

from model import ViT
from probe import worker_probe_transforms
from interpret.datasets import load_dataset
from interpret.effective_rank import EffectiveRank


# -------------------------------------------------------------------------------------------------
# REPRESENTATION EXTRACTION
# -------------------------------------------------------------------------------------------------

# The token groups every hook MUST provide; the rig validates and stores only these. Downstream methods need the
# patch tokens (spatial maps, GAP) and the CLS token (global summary). Any other group a hook writes is ignored.
REQUIRED_GROUPS = ("patch", "cls")


# Validates the hook contract documented in interpret/README.md. Add new checks here whenever a new contract
# property is identified, and document it there too.
def assert_hook_interface(storage_dict, layer_names, batch_size, global_state):
    assert len(layer_names) == len(set(layer_names)), "Duplicate layer keys in returned list."

    for layer in layer_names:
        where_layer = f"Layer '{layer}'"
        assert layer in storage_dict, f"{where_layer} missing from hook output."
        for group_name in REQUIRED_GROUPS:
            assert group_name in storage_dict[layer], f"{where_layer} missing required group '{group_name}'."
        group_dict = {name: storage_dict[layer][name] for name in REQUIRED_GROUPS}  # extra groups are ignored

        for group_name, tensor in group_dict.items():
            where = f"{where_layer} group '{group_name}'"
            assert tensor.ndim == 3, f"{where} not 3D."
            assert tensor.shape[0] == batch_size, f"{where} batch {tensor.shape[0]} != {batch_size}."
            assert tensor.shape[1] > 0, f"{where} is empty (0 tokens); omit empty groups."
            assert not tensor.requires_grad, f"{where} not detached."
            assert torch.isfinite(tensor).all(), f"{where} contains non-finite values."

        patch, cls = group_dict["patch"], group_dict["cls"]
        assert cls.shape[1] == 1, f"{where_layer} 'cls' group has {cls.shape[1]} tokens, expected exactly 1."
        assert math.isqrt(patch.shape[1]) ** 2 == patch.shape[1], f"{where_layer} 'patch' length {patch.shape[1]} not a perfect square."
        assert patch.shape[2] == cls.shape[2], f"{where_layer} hidden dims differ: patch {patch.shape[2]} vs cls {cls.shape[2]}."

        # Token counts are fixed by the first layer of the first batch and must then match for ALL layers and batches.
        tokens = {name: tensor.shape[1] for name, tensor in group_dict.items()}
        assert tokens == global_state.setdefault("tokens", tokens), f"{where_layer} token counts {tokens} mismatch global {global_state['tokens']}."


# Slices one sample out of the batch for the kept layers (devices are main()'s job). copy=True gives each sample
# its own storage, since saving a view would serialize the whole batch.
def _process_single_sample_features(storage_dict, kept_layer_names, batch_idx):
    return {
        layer: {group: storage_dict[layer][group][batch_idx].to(torch.float32, copy=True) for group in REQUIRED_GROUPS}
        for layer in kept_layer_names
    }


# Samples tiles (stratified, seeded), runs the model with its hooks, checks the hook contract, keeps every layer_stride-th
# layer and writes chunk_N.pt files plus metadata.json (last, so it marks a complete run) to output_dir. `settings` is what
# determines the chunks (hashed by main(), recorded in metadata.json); `extract_cfg` holds knobs that don't change them.
# No device moves here: the model arrives on its device and `transform` already outputs tensors on it.
def extract_features_to_chunks(model, dataset, transform, settings, extract_cfg, output_dir):
    # Only a limit (num_samples below the dataset size) makes this a real subsample; otherwise every sample is taken.
    # Stratified over the dataset's configured stratification dimension (see datasets.py): shuffle the samples of each
    # value, rank them by position inside it, then take the lowest ranks first. Every value contributes equally (one
    # that runs dry gives up all it has), and the draw is deterministic given the seed.
    rng, groups = random.Random(settings["seed"]), {}
    for i, value in enumerate(dataset.stratification_values):
        groups.setdefault(value, []).append(i)
    ranked = []
    for members in groups.values():
        rng.shuffle(members)
        ranked += [(rank, rng.random(), i) for rank, i in enumerate(members)]
    sampled_indices = [i for _, _, i in sorted(ranked)[:settings["num_samples"]]]
    subset = torch.utils.data.Subset(dataset, sampled_indices)
    # The workers only decode the images; transform and model run in this process (the transform ends with a CUDA move).
    loader = torch.utils.data.DataLoader(subset, batch_size=extract_cfg["batch_size"], shuffle=False, num_workers=extract_cfg["num_workers"], collate_fn=lambda batch: tuple(zip(*batch)))

    output_dir.mkdir(parents=True)

    storage_dict = {}
    layer_names = model.register_interpretability_hooks(storage_dict)
    kept_layer_names = layer_names[::-settings["layer_stride"]][::-1]

    interface_state = {}

    chunk_idx = 0
    current_chunk_data = []

    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_size = len(batch_images)
            processed_images = torch.stack([transform(image) for image in batch_images])

            storage_dict.clear()  # the hooks hold a reference to this dict, so clear it in place
            model(processed_images)

            assert_hook_interface(storage_dict, layer_names, batch_size, interface_state)  # on ALL layers, before the stride

            for b in range(batch_size):
                sample_features = _process_single_sample_features(storage_dict, kept_layer_names, b)

                # Saved sample format, read back by ChunkedFeatureDataset: image = the model input (clone: a view would
                # keep the whole batch), label = the dataset's metadata dict, layers = {layer: {group: (T, C) float32}}.
                datapoint = {
                    "image": processed_images[b].clone(),
                    "label": batch_labels[b],
                    "layers": sample_features
                }
                current_chunk_data.append(datapoint)

                if len(current_chunk_data) == extract_cfg["chunk_size"]:
                    torch.save(current_chunk_data, output_dir / f"chunk_{chunk_idx}.pt")
                    chunk_idx += 1
                    current_chunk_data = []

        # The last, partial chunk.
        if current_chunk_data:
            torch.save(current_chunk_data, output_dir / f"chunk_{chunk_idx}.pt")

    metadata = {
        **settings,
        # Exact code version that made these chunks; a -dirty suffix means uncommitted changes.
        "git_commit": subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent, text=True).strip(),
        "chunk_size": extract_cfg["chunk_size"],
        "kept_layers": kept_layer_names,
        "sampled_dataset_indices": sampled_indices,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata))


# -------------------------------------------------------------------------------------------------
# DATASET CLASSES
# -------------------------------------------------------------------------------------------------

# Serves the samples saved by extract_features_to_chunks as (image, label, layers): sample idx is at position
# idx % chunk_size of chunk_{idx // chunk_size}.pt, so only chunk_size and the sample count are read from metadata.json.
class ChunkedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        metadata = json.loads((self.chunks_dir / "metadata.json").read_text())
        self.chunk_size, self.num_samples = metadata["chunk_size"], len(metadata["sampled_dataset_indices"])
        self.current_chunk_idx = None
        self.current_chunk_data = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk_idx, position = divmod(idx, self.chunk_size)

        # One-chunk cache: methods iterate the samples sequentially.
        if self.current_chunk_idx != chunk_idx:
            self.current_chunk_data = torch.load(self.chunks_dir / f"chunk_{chunk_idx}.pt", weights_only=False)
            self.current_chunk_idx = chunk_idx

        datapoint = self.current_chunk_data[position]
        return datapoint["image"], datapoint["label"], datapoint["layers"]


# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

# The model's own eval preprocessing: probe.py's transform plus the normalization it applies at its model-call sites.
def evaluation_preprocess(train_cfg):
    transform, _ = worker_probe_transforms(train_cfg)
    # Normalization is part of probe.py's preprocessing, but it is applied at probe model-call
    # sites rather than inside worker_probe_transforms(). Since probe.py must not be edited, we
    # reimplement it here. Regularly compare the pipelines to prevent drift (e.g. with an agent).
    mean = torch.tensor(train_cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(train_cfg["data"]["std"]).view(3, 1, 1)
    return transforms.Compose([transform, transforms.Lambda(lambda x: (x - mean) / std)])


# Builds the evaluated model from a Nanopath checkpoint dict (same weight-key choice as probe.py).
# To interpret another model, swap this function; the rig only needs `register_interpretability_hooks`.
def load_model(checkpoint, model_weights):
    model = ViT(variant=checkpoint["config"]["model"]["type"])
    model.load_state_dict(checkpoint[{"ema": "model_ema", "model": "model"}[model_weights]], strict=True)
    return model.eval()


# DEVICE PRINCIPLE: main() is the only place that moves tensors or modules between devices. Every other
# function computes on whatever device its inputs already live on, so there is exactly one place to read
# to see who is where.
def main():
    # Usage (from repo root, so `model`/`probe` import): python -m interpret.interpret configs/main.yaml
    # (expandvars resolves `$USER`, as in train.py).
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    icfg = cfg["interpret"]
    checkpoint = torch.load(icfg["checkpoint_path"], map_location="cpu", weights_only=False)
    model = load_model(checkpoint, icfg["model_weights"]).to("cuda")
    # Eval preprocessing comes from the checkpoint's own training config; the final Lambda puts each
    # preprocessed image on the GPU so extract_features_to_chunks never has to move anything itself.
    transform = transforms.Compose([evaluation_preprocess(checkpoint["config"]), transforms.Lambda(lambda x: x.to("cuda"))])
    print(f"Loaded checkpoint from {icfg['checkpoint_path']}")

    # One dataset per run: `dataset` names it, `datasets` holds each identifier's path and stratification dimension.
    dataset_cfg = icfg["datasets"][icfg["dataset"]]
    dataset = load_dataset(icfg["dataset"], dataset_cfg)

    # Output layout (see README.md): output_dir/<hash of the settings below>/{chunks/, results_<methods>.json}. An existing
    # hash subdirectory is reused as is; if its run died, chunks/metadata.json is missing and loading it below crashes,
    # so delete that subdirectory by hand and rerun.
    extract_cfg = icfg["extract"]
    settings = {
        "checkpoint_path": icfg["checkpoint_path"], "checkpoint_step": checkpoint["step"], "model_weights": icfg["model_weights"],
        "dataset": icfg["dataset"], "dataset_config": dataset_cfg,
        "seed": extract_cfg["seed"], "num_samples": extract_cfg["num_samples"], "layer_stride": extract_cfg["layer_stride"],
    }
    run_dir = Path(icfg["output_dir"]) / hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    if run_dir.exists():
        print(f"Reusing extraction in {run_dir}")
    else:
        print(f"Extracting features into {run_dir}...")
        extract_features_to_chunks(model, dataset, transform, settings, extract_cfg, run_dir / "chunks")

    # Each method is a class built as Method(data, its YAML sub-block); run() returns a results dict.
    # Track A methods read the saved chunks; a Track B method would be handed the model/loader instead.
    chunks = ChunkedFeatureDataset(run_dir / "chunks")
    results = {}
    for name in icfg["methods"]:
        print(f"Running {name}...")
        if name == "effective_rank":
            results[name] = EffectiveRank(chunks, icfg[name]).run()
        else:
            raise KeyError(f"Unknown method '{name}'; add it to the method dispatch in main() of interpret/interpret.py")
    # The file name lists the methods that ran, so runs with different method subsets don't overwrite each other.
    (run_dir / f"results_{'_'.join(sorted(icfg['methods']))}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
