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
        assert layer in storage_dict, f"Layer '{layer}' missing from hook output."
        for group_name in REQUIRED_GROUPS:
            assert group_name in storage_dict[layer], f"Layer '{layer}' missing required group '{group_name}'."
        group_dict = {name: storage_dict[layer][name] for name in REQUIRED_GROUPS}  # extra groups are ignored

        # Token counts of the required groups are fixed by the first layer of the first batch and must
        # then match for ALL layers and batches.
        current_tokens = {name: tensor.shape[1] for name, tensor in group_dict.items()}
        if global_state["expected_tokens"] is None:
            global_state["expected_tokens"] = current_tokens
        expected_tokens = global_state["expected_tokens"]
        assert current_tokens == expected_tokens, f"Layer '{layer}' groups/token counts {current_tokens} mismatch global {expected_tokens}."

        layer_expected_dim = None
        for group_name, tensor in group_dict.items():
            assert tensor.ndim == 3, f"Layer '{layer}' group '{group_name}' not 3D."
            batch, tokens, hidden_dim = tensor.shape
            assert batch == batch_size, f"Layer '{layer}' group '{group_name}' batch {batch} != {batch_size}."
            assert tokens > 0, f"Layer '{layer}' group '{group_name}' is empty (0 tokens); omit empty groups."
            if group_name == "cls":
                assert tokens == 1, f"Layer '{layer}' 'cls' group has {tokens} tokens, expected exactly 1."

            if layer_expected_dim is None:
                layer_expected_dim = hidden_dim
            assert hidden_dim == layer_expected_dim, (
                f"Layer '{layer}' group '{group_name}' dim {hidden_dim} != "
                f"expected {layer_expected_dim} for this layer."
            )

            assert not tensor.requires_grad, f"Layer '{layer}' group '{group_name}' not detached."
            assert torch.isfinite(tensor).all(), f"Layer '{layer}' group '{group_name}' contains non-finite values."

            if group_name == "patch":
                sqrt_tokens = math.isqrt(tokens)
                assert sqrt_tokens * sqrt_tokens == tokens, (
                    f"Layer '{layer}' 'patch' length {tokens} not a perfect square."
                )


# Slices one sample out of the batch for the kept layers (devices are main()'s job). copy=True gives each sample
# its own storage, since saving a view would serialize the whole batch.
def _process_single_sample_features(storage_dict, kept_layer_names, batch_idx):
    return {
        layer: {group: storage_dict[layer][group][batch_idx].to(torch.float32, copy=True) for group in REQUIRED_GROUPS}
        for layer in kept_layer_names
    }


# Samples tiles (stratified, seeded), runs the model with its hooks, checks the hook contract, keeps every
# layer_stride-th layer and writes chunk_N.pt files plus paths.json (last, so it marks a complete run) to output_dir.
# cfg is the top-level `interpret` config. No device moves here: the model arrives on its device and `transform`
# already outputs tensors on it.
def extract_features_to_chunks(model, dataset, transform, cfg, output_dir):
    pcfg = cfg["preprocess"]
    # Sample stratified over the dataset's configured stratification dimension (see datasets.py): shuffle the samples
    # of each value, rank them by position inside it, then take the lowest ranks first. Every value contributes
    # equally (one that runs dry gives up all it has), and the draw is deterministic given the seed.
    rng, groups = random.Random(cfg["seed"]), {}
    for i, value in enumerate(dataset.stratification_values):
        groups.setdefault(value, []).append(i)
    ranked = []
    for members in groups.values():
        rng.shuffle(members)
        ranked += [(rank, rng.random(), i) for rank, i in enumerate(members)]
    sampled_indices = [i for _, _, i in sorted(ranked)[:pcfg["num_samples"]]]
    subset = torch.utils.data.Subset(dataset, sampled_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=pcfg["batch_size"], shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

    output_dir.mkdir(parents=True)

    storage_dict = {}
    layer_names = model.register_interpretability_hooks(storage_dict)
    kept_layer_names = layer_names[::-pcfg["layer_stride"]][::-1]

    # Fixed by the first batch, then enforced on every layer and batch.
    interface_state = {"expected_tokens": None}

    chunk_idx = 0
    current_chunk_data = []
    sample_mapping = []

    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_size = len(batch_images)
            processed_images = torch.stack([transform(image) for image in batch_images])

            storage_dict.clear()  # the hooks hold a reference to this dict, so clear it in place
            model(processed_images)

            assert_hook_interface(storage_dict, layer_names, batch_size, interface_state)  # on ALL layers, before the stride

            for b in range(batch_size):
                sample_features = _process_single_sample_features(storage_dict, kept_layer_names, b)

                datapoint = {
                    "image": processed_images[b],
                    "label": batch_labels[b],
                    "layers": sample_features
                }
                current_chunk_data.append(datapoint)

                chunk_path = output_dir / f"chunk_{chunk_idx}.pt"
                sample_mapping.append({
                    "original_idx": sampled_indices[len(sample_mapping)],
                    "chunk_path": str(chunk_path),
                    "index_in_chunk": len(current_chunk_data) - 1
                })

                if len(current_chunk_data) >= pcfg["chunk_size"]:
                    torch.save(current_chunk_data, chunk_path)
                    chunk_idx += 1
                    current_chunk_data = []

        # The last, partial chunk (chunk_path still names it).
        if current_chunk_data:
            torch.save(current_chunk_data, chunk_path)

    paths_data = {
        # Exact code version that made these chunks; a -dirty suffix means uncommitted changes.
        "git_commit": subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent, text=True).strip(),
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "num_samples": pcfg["num_samples"],
        "checkpoint_path": cfg["checkpoint_path"],
        "model_weights": cfg["model_weights"],
        "layer_stride": pcfg["layer_stride"],
        "kept_layers": kept_layer_names,
        "sampled_dataset_indices": sampled_indices,
        "sample_mapping": sample_mapping
    }
    (output_dir / "paths.json").write_text(json.dumps(paths_data))


# -------------------------------------------------------------------------------------------------
# DATASET CLASSES
# -------------------------------------------------------------------------------------------------

# Serves the samples saved by extract_features_to_chunks (found through paths.json) as (image, label, layers).
class ChunkedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, paths_file):
        metadata = json.loads(Path(paths_file).read_text())
        self.sample_mapping = metadata["sample_mapping"]
        self.current_chunk_path = None
        self.current_chunk_data = None

    def __len__(self):
        return len(self.sample_mapping)

    def __getitem__(self, idx):
        mapping = self.sample_mapping[idx]
        chunk_path = mapping["chunk_path"]

        # One-chunk cache: methods iterate the samples sequentially.
        if self.current_chunk_path != chunk_path:
            self.current_chunk_data = torch.load(chunk_path, weights_only=False)
            self.current_chunk_path = chunk_path

        datapoint = self.current_chunk_data[mapping["index_in_chunk"]]
        return datapoint["image"], datapoint["label"], datapoint["layers"]


# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

# The model's own eval preprocessing: probe.py's transform plus the normalization it applies at its model-call sites.
def evaluation_preprocess(cfg):
    transform, _ = worker_probe_transforms(cfg)
    # Normalization is part of probe.py's preprocessing, but it is applied at probe model-call
    # sites rather than inside worker_probe_transforms(). Since probe.py must not be edited, we
    # reimplement it here. Regularly compare the pipelines to prevent drift (e.g. with an agent).
    mean = torch.tensor(cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["data"]["std"]).view(3, 1, 1)
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

    # Layout: output_dir/<hash of everything that determines the chunks>/{chunks/ (chunk_N.pt + paths.json), results_<methods>.json}.
    # An existing hash subdirectory is reused as is; if its run died, paths.json (written last) is missing and loading it
    # below crashes, so delete that subdirectory by hand and rerun.
    settings = {
        "checkpoint_path": icfg["checkpoint_path"], "checkpoint_step": checkpoint["step"], "model_weights": icfg["model_weights"],
        "dataset": icfg["dataset"], "dataset_config": dataset_cfg, "seed": icfg["seed"],
        "num_samples": icfg["preprocess"]["num_samples"], "layer_stride": icfg["preprocess"]["layer_stride"],
    }
    run_dir = Path(icfg["output_dir"]) / hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    if run_dir.exists():
        print(f"Reusing extraction in {run_dir}")
    else:
        print(f"Extracting features into {run_dir}...")
        extract_features_to_chunks(model, dataset, transform, icfg, run_dir / "chunks")

    # Each method is a class built as Method(data, its YAML sub-block); run() returns a results dict.
    # Track A methods read the saved chunks; a Track B method would be handed the model/loader instead.
    chunks = ChunkedFeatureDataset(run_dir / "chunks" / "paths.json")
    results = {}
    for name in icfg["methods"]:
        print(f"Running {name}...")
        if name == "effective_rank":
            results[name] = EffectiveRank(chunks, icfg[name]).run()
    # The file name lists the methods that ran, so runs with different method subsets don't overwrite each other.
    (run_dir / f"results_{'_'.join(sorted(icfg['methods']))}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
