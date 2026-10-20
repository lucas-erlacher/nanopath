# interpret/interpret.py: extracts hooked activations to chunks, then runs the configured methods (see README.md).
import os
import sys
import math
import json
import hashlib
import subprocess
import yaml
import torch
from sklearn.utils import resample
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from model import ViT
from probe import worker_probe_transforms
from interpret.datasets import load_dataset
from interpret.effective_rank import EffectiveRank


# -------------------------------------------------------------------------------------------------
# REPRESENTATION EXTRACTION
# -------------------------------------------------------------------------------------------------

# Token groups every hook must provide; only these are validated and stored, any other group is ignored.
REQUIRED_GROUPS = ("patch", "cls")


# Validates the hook contract in interpret/README.md; add new checks here and document them there.
def assert_hook_interface(storage_dict, layer_names, batch_size, global_state):
    assert len(layer_names) == len(set(layer_names)), "Duplicate layer keys in returned list."

    for layer in layer_names:
        where_layer = f"Layer '{layer}'"
        assert layer in storage_dict, f"{where_layer} missing from hook output."
        for group_name in REQUIRED_GROUPS:
            assert group_name in storage_dict[layer], f"{where_layer} missing required group '{group_name}'."
        group_dict = {name: storage_dict[layer][name] for name in REQUIRED_GROUPS}

        for group_name, tensor in group_dict.items():
            where = f"{where_layer} group '{group_name}'"
            assert tensor.ndim == 3, f"{where} not 3D."
            assert tensor.shape[0] == batch_size, f"{where} batch {tensor.shape[0]} != {batch_size}."
            assert tensor.shape[1] > 0, f"{where} is empty (0 tokens); omit empty groups."
            assert not tensor.requires_grad, f"{where} not detached."
            assert torch.isfinite(tensor).all(), f"{where} contains non-finite values."

        patch = group_dict["patch"]
        cls = group_dict["cls"]
        assert cls.shape[1] == 1, f"{where_layer} 'cls' group has {cls.shape[1]} tokens, expected exactly 1."
        assert math.isqrt(patch.shape[1]) ** 2 == patch.shape[1], f"{where_layer} 'patch' length {patch.shape[1]} not a perfect square."
        assert patch.shape[2] == cls.shape[2], f"{where_layer} hidden dims differ: patch {patch.shape[2]} vs cls {cls.shape[2]}."

        # Token counts must match across all layers and batches (the first one sets them).
        tokens = {name: tensor.shape[1] for name, tensor in group_dict.items()}
        if "tokens" not in global_state:
            global_state["tokens"] = tokens
        assert tokens == global_state["tokens"], f"{where_layer} token counts {tokens} mismatch the first layer's {global_state['tokens']}."


# Samples tiles (stratified, seeded), runs the model, checks the hook contract and writes chunk_N.pt files plus
# metadata.json (last, so it marks a complete run). `settings` determines the chunks (hashed in main, saved in metadata.json).
def extract_features_to_chunks(model, dataset, transform, settings, extract_cfg, output_dir):
    # Seeded subsample that keeps the proportions of the dataset's stratification dimension (a metadata field).
    dimension = settings["dataset_config"]["stratification_dimension"]
    values = [metadata[dimension] for _, metadata in dataset.records]
    sampled_indices = resample(list(range(len(dataset))), n_samples=settings["num_samples"], replace=False, stratify=values, random_state=settings["seed"])
    subset = torch.utils.data.Subset(dataset, sampled_indices)
    # "fork" as in probe.py: Python 3.14 defaults to "forkserver", which pickles the dataset and the collate lambda to every worker.
    loader = torch.utils.data.DataLoader(
        subset, batch_size=extract_cfg["batch_size"], shuffle=False, num_workers=extract_cfg["num_workers"],
        multiprocessing_context="fork", collate_fn=lambda batch: tuple(zip(*batch)),
    )

    output_dir.mkdir(parents=True)

    storage_dict = {}
    layer_names = model.register_interpretability_hooks(storage_dict)
    # Keep every layer_stride-th layer, counting from the last one (so the last is always kept).
    kept_layer_names = []
    for steps_from_last, layer in enumerate(reversed(layer_names)):
        if steps_from_last % settings["layer_stride"] == 0:
            kept_layer_names.append(layer)
    kept_layer_names.reverse()  # back to forward-pass order

    interface_state = {}  # remembers the token counts of the first layer of the first batch, which all later ones must match

    chunk_idx = 0
    chunk_samples = []

    with torch.inference_mode():
        for batch_images, batch_metadata in tqdm(loader, desc="Extracting"):
            batch_size = len(batch_images)
            processed_images = torch.stack([transform(image) for image in batch_images])

            storage_dict.clear()  # the hooks hold this dict, so clear it in place
            model(processed_images)

            assert_hook_interface(storage_dict, layer_names, batch_size, interface_state)  # before the stride

            for b in range(batch_size):
                # One sample's kept layers; copy=True because saving a view would serialize the whole batch.
                sample_layers = {}
                for layer in kept_layer_names:
                    sample_layers[layer] = {}
                    for group in REQUIRED_GROUPS:
                        sample_layers[layer][group] = storage_dict[layer][group][b, :, :].to(torch.float32, copy=True)

                # Sample format (read by ChunkedFeatureDataset): image, metadata (from the dataset), layers {layer: {group: (T, C)}};
                # the image is cloned because a view would keep the whole batch alive.
                sample = {
                    "image": processed_images[b, :, :, :].clone(),
                    "metadata": batch_metadata[b],
                    "layers": sample_layers
                }
                chunk_samples.append(sample)

                if len(chunk_samples) == extract_cfg["chunk_size"]:
                    torch.save(chunk_samples, output_dir / f"chunk_{chunk_idx}.pt")
                    chunk_idx += 1
                    chunk_samples = []

        # The last, partial chunk.
        if chunk_samples:
            torch.save(chunk_samples, output_dir / f"chunk_{chunk_idx}.pt")

    metadata = {
        **settings,
        # Exact code version; -dirty means uncommitted changes.
        "git_commit": subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent, text=True).strip(),
        "chunk_size": extract_cfg["chunk_size"],
        "kept_layers": kept_layer_names,
        "sampled_dataset_indices": sampled_indices,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata))


# -------------------------------------------------------------------------------------------------
# DATASET CLASSES
# -------------------------------------------------------------------------------------------------

# Serves saved samples as (image, metadata, layers) on `device`: sample idx is at idx % chunk_size of chunk_{idx // chunk_size}.pt.
class ChunkedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = Path(data_dir)
        self.device = device
        metadata = json.loads((self.data_dir / "metadata.json").read_text())
        self.chunk_size = metadata["chunk_size"]
        self.num_samples = len(metadata["sampled_dataset_indices"])
        self.current_chunk_idx = None
        self.current_chunk_samples = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        position = idx % self.chunk_size

        # One-chunk cache: methods iterate the samples sequentially.
        if self.current_chunk_idx != chunk_idx:
            self.current_chunk_samples = torch.load(self.data_dir / f"chunk_{chunk_idx}.pt", map_location=self.device, weights_only=False)
            self.current_chunk_idx = chunk_idx

        sample = self.current_chunk_samples[position]
        return sample["image"], sample["metadata"], sample["layers"]


# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

# probe.py's eval transform plus the normalization it applies at its model-call sites; 
# duplicated here because probe.py is locked, so keep the two in sync. 
def evaluation_preprocess(train_cfg):
    transform, _ = worker_probe_transforms(train_cfg)
    mean = torch.tensor(train_cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(train_cfg["data"]["std"]).view(3, 1, 1)
    return transforms.Compose([transform, transforms.Lambda(lambda x: (x - mean) / std)])


# Loads a Nanopath checkpoint and returns the model (eval mode, on the CPU), the training config its preprocessing comes
# from, and its training step (part of the cache hash). To interpret another model, swap this function.
def load_model(checkpoint_path, model_weights):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ViT(variant=checkpoint["config"]["model"]["type"])
    model.load_state_dict(checkpoint[model_weights], strict=True)
    return model.eval(), checkpoint["config"], checkpoint["step"]


# Everything computes on ONE device (the GPU); main() is the only place that moves tensors or modules between
# devices, every other function computes wherever its inputs already live.
def main():

    # ---- SETUP ----
    device = "cuda"
    # Usage (from the repo root): python -m interpret.interpret configs/main.yaml
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    icfg = cfg["interpret"]
    model, train_cfg, checkpoint_step = load_model(icfg["checkpoint_path"], icfg["model_weights"])
    model = model.to(device)
    # The final Lambda puts each image on `device`, so the extractor never moves anything itself.
    transform = transforms.Compose([evaluation_preprocess(train_cfg), transforms.Lambda(lambda x: x.to(device))])
    print(f"Loaded checkpoint from {icfg['checkpoint_path']}")

    dataset_cfg = icfg["datasets"][icfg["dataset"]]
    dataset = load_dataset(icfg["dataset"], cfg["probe"]["dataset_roots"])

    # Extractions are cached in output_dir/data/<hash of settings> (layout in README.md); if a run died, metadata.json is
    # missing and loading crashes below: delete that directory and rerun.
    extract_cfg = icfg["extract"]
    settings = {
        "checkpoint_path": icfg["checkpoint_path"], "checkpoint_step": checkpoint_step, "model_weights": icfg["model_weights"],
        "dataset": icfg["dataset"], "dataset_config": dataset_cfg,
        "seed": extract_cfg["seed"], "num_samples": extract_cfg["num_samples"], "layer_stride": extract_cfg["layer_stride"],
    }
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    data_dir = Path(icfg["output_dir"]) / "data" / settings_hash

    # ---- EXTRACTION ----
    if data_dir.exists():
        print(f"Reusing extraction in {data_dir}")
    else:
        print(f"Extracting features into {data_dir}...")
        extract_features_to_chunks(model, dataset, transform, settings, extract_cfg, data_dir)

    chunked_dataset = ChunkedFeatureDataset(data_dir, device)
    
    # ---- ANALYSIS ----
    # One results folder per extraction and set of methods, so other extractions or method subsets don't overwrite it.
    sorted_method_names = sorted(icfg["methods"])
    methods_label = "_".join(sorted_method_names)
    results_dir = Path(icfg["output_dir"]) / "results" / f"{settings_hash}_{methods_label}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in icfg["methods"]:
        print(f"Running {name}...")
        if name == "effective_rank":
            method = EffectiveRank(chunked_dataset, icfg[name])
        else:
            raise KeyError(f"Unknown method '{name}'; add it to the method dispatch in main() of interpret/interpret.py")
        method.run()
        results[name] = method.results

        # Each method saves the plots it defines into its own subfolder.
        plot_dir = results_dir / name
        plot_dir.mkdir(exist_ok=True)
        for plot_name, figure in method.plot().items():
            figure.savefig(plot_dir / f"{plot_name}.png", bbox_inches="tight")

    results_json = json.dumps(results, indent=2)
    (results_dir / "results.json").write_text(results_json)
    print(results_json)


if __name__ == "__main__":
    main()
