# interpret/interpret.py: extracts hooked activations to files, then runs the configured methods on them (see README.md).
import os
import sys
import math
import random
import json
import hashlib
import time
import subprocess
import yaml
import numpy as np
import torch
from sklearn.utils import resample
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from model import ViT
from probe import worker_probe_transforms
from interpret.datasets import load_dataset
from interpret.utils import write_dashboard, TIMINGS

# Headless plotting (figures are only saved to files): Colab exports MPLBACKEND=<notebook backend>, which is not installed in the venv.
os.environ["MPLBACKEND"] = "agg"
from interpret.methods.effective_rank import EffectiveRank
from interpret.methods.cka import CKA
from interpret.methods.linear_probe import LinearProbe


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


# Samples tiles (stratified, seeded) and writes the extraction (layout in README.md): the labels, then one pass of the model over the tiles that collects the activations and the images.
# info.json is written last, so it marks a complete run. `settings` determines the files (hashed in main, saved in info.json).
def extract_activations(model, dataset, transform, settings, extract_cfg, output_dir):
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
    num_samples = len(sampled_indices)

    (output_dir / "activations").mkdir(parents=True)
    (output_dir / "precomputed").mkdir()

    # LABELS: every metadata field as integers (its sorted unique values become 0, 1, 2, ...) plus the names to map them back. Row i is the i-th sampled tile, like in every other file.
    sampled_metadata = [dataset.records[i][1] for i in sampled_indices]
    labels = {}  # e.g. {"medical_center": tensor([0, 1, 1, 0, ...])} - each tensor holds one integer per datapoint (int representation of metadata strings)
    label_names = {}  # e.g. {"medical_center": ["center_a", "center_b"]} - this maps the integers back to human readable strings
    for field in sampled_metadata[0]:
        label_names[field] = sorted({metadata[field] for metadata in sampled_metadata})
        value_to_integer = {value: integer for integer, value in enumerate(label_names[field])}
        labels[field] = torch.tensor([value_to_integer[metadata[field]] for metadata in sampled_metadata])
    torch.save(labels, output_dir / "labels.pt")
    (output_dir / "label_names.json").write_text(json.dumps(label_names))

    storage_dict = {}
    layer_names = model.register_interpretability_hooks(storage_dict)
    # Keep every layer_stride-th layer, counting from the last one (so the last is always kept).
    kept_layer_names = []
    for steps_from_last, layer in enumerate(reversed(layer_names)):
        if steps_from_last % settings["layer_stride"] == 0:
            kept_layer_names.append(layer)
    kept_layer_names.reverse()  # back to forward-pass order

    interface_state = {}  # remembers the token counts of the first layer of the first batch, which all later ones must match

    # EXTRACTION PASS: the tensors of every kept layer and the images are allocated at full size in RAM after the first batch (no double peak from concatenating), each batch fills in its rows, and everything is saved at the end.
    # The patch mean is computed here from the float32 tokens, so the methods never average tokens.
    tensors = {}
    start = 0  # the first row of the current batch
    with torch.inference_mode():
        for batch_images, _ in tqdm(loader, desc="Extracting"):
            batch_size = len(batch_images)
            processed_images = torch.stack([transform(image) for image in batch_images])  # (batch, 3, H, W)
            storage_dict.clear()  # the hooks hold this dict, so clear it in place
            model(processed_images)

            assert_hook_interface(storage_dict, layer_names, batch_size, interface_state)  # before the stride

            rows = slice(start, start + batch_size)
            # store images
            if start == 0:  # allocate the image tensor with the first batch (we only know dimensions at runtime)
                num_channels = processed_images.shape[1]
                height = processed_images.shape[2]
                width = processed_images.shape[3]
                images = torch.empty(num_samples, num_channels, height, width)
            images[rows] = processed_images
            
            for layer in kept_layer_names:
                patch = storage_dict[layer]["patch"]  # (batch, seq_len, channel_dim)
                if layer not in tensors:  # initialize tensors for layer if not yet initialized
                    seq_len = patch.shape[1]
                    channel_dim = patch.shape[2]
                    tensors[layer] = {
                        "patch_tokens": torch.empty(num_samples, seq_len, channel_dim),
                        "cls": torch.empty(num_samples, channel_dim),
                        "patch_mean": torch.empty(num_samples, channel_dim),
                    }
                # store patch data
                tensors[layer]["patch_tokens"][rows] = patch
                tensors[layer]["patch_mean"][rows] = patch.mean(dim=1)  # mean over the patch tokens
                # store CLS data
                tensors[layer]["cls"][rows] = storage_dict[layer]["cls"][:, 0, :]  # the CLS group is (batch, 1, C), indexing out position 0 gives plain C-vectors
            start += batch_size
    assert start == num_samples

    for layer, layer_tensors in tensors.items():
        torch.save(layer_tensors["patch_tokens"], output_dir / "activations" / f"{layer}_patch_tokens.pt")
        torch.save(layer_tensors["cls"], output_dir / "activations" / f"{layer}_cls.pt")
        torch.save(layer_tensors["patch_mean"], output_dir / "precomputed" / f"{layer}_patch_mean.pt")
    torch.save(images, output_dir / "images.pt")

    info = {
        **settings,
        # Exact code version; -dirty means uncommitted changes.
        "git_commit": subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent, text=True).strip(),
        "kept_layers": kept_layer_names,
        "sampled_dataset_indices": sampled_indices,
    }
    (output_dir / "info.json").write_text(json.dumps(info))


# -------------------------------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------------------------------

# One layer of an extraction (layout in README.md): row i of every file is the i-th sampled tile. The constructor sends nothing to `device`: the tensors are CPU tensors in memory
# (the big files `patch_tokens` and `image` are memory-mapped and only opened when enabled), all of them are private, and the two ways out, get_block (a whole small block) and loader (batches), are the only
# places that move data to the device; __getitem__ returns one example as CPU tensors. A part that is not enabled is an empty tensor, so items always have the same keys.
# Only two facts are public: label_names and vector_types.
class ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, layer_name, device, patch_tokens=False, image=False):
        data_dir = Path(data_dir)
        self._device = device
        # load data that the dataset spans
        # the activations of this layer for every example: the vectors, (N, C) each, are in memory, the big patch tokens, (N, 256, C), are memory-mapped and None unless enabled
        self._activations = {
            "cls": torch.load(data_dir / "activations" / f"{layer_name}_cls.pt"),
            "patch_tokens": torch.load(data_dir / "activations" / f"{layer_name}_patch_tokens.pt", mmap=True) if patch_tokens else None,
            "patch_mean": torch.load(data_dir / "precomputed" / f"{layer_name}_patch_mean.pt"),
        }
        self._images = torch.load(data_dir / "images.pt", mmap=True) if image else None  # (N, 3, H, W)
        self._labels = torch.load(data_dir / "labels.pt")  # e.g. {"medical_center": tensor([0, 1, 1, 0, ...])} - each tensor holds one integer per datapoint (int representation of metadata strings)
        # the two public facts
        self.label_names = json.loads((data_dir / "label_names.json").read_text())  # e.g. {"medical_center": ["center_a", "center_b"]} - this maps the integers back to human readable strings
        self.vector_types = {"cls": "CLS token", "patch_mean": "mean over the patch tokens"}  # the vectors every item holds, with a description for titles

    def __len__(self):
        return len(self._activations["cls"])

    # a whole small block of this layer on the device: `field` names a dict of this class ("activations" or "labels") and `key` picks the block in it, e.g. get_block("activations", "cls") or get_block("labels", "medical_center");
    # without a key the whole dict is returned, which only exists for the labels (get_block("labels", to_device=False), a dict has no device move).
    # The big blocks (`patch_tokens`, `images`) are deliberately not available here: they are memory-mapped so that they are never loaded whole, and are only read in batches through loader.
    # to_device=False skips the device move, for the few callers that work on the host (sklearn and printouts).
    def get_block(self, field, key=None, to_device=True):
        assert field != "images" and key != "patch_tokens" and (field, key) != ("activations", None), f"'{field}' / '{key}' includes a big block: it is memory-mapped so that it is never loaded whole, so it can only be read in batches through loader"
        block = getattr(self, f"_{field}")
        if key is not None:
            block = block[key]
        return block.to(self._device) if to_device else block

    # one example as CPU tensors (this is what DataLoader and Subset fetch); the device move happens per batch in loader, not per example here
    def __getitem__(self, i):
        t = time.perf_counter()  # TIMING
        not_loaded = torch.empty(0)
        image = self._images[i] if self._images is not None else not_loaded
        patch_tokens = self._activations["patch_tokens"][i] if self._activations["patch_tokens"] is not None else not_loaded
        item = {
            "image": image,
            "cls": self._activations["cls"][i],
            "patch_tokens": patch_tokens,
            "patch_mean": self._activations["patch_mean"][i],
        }
        TIMINGS["getitem tensors"] += time.perf_counter() - t  # TIMING
        t = time.perf_counter()  # TIMING
        item["labels"] = {field: values[i] for field, values in self._labels.items()}
        TIMINGS["getitem labels"] += time.perf_counter() - t  # TIMING
        return item

    # batches of the examples at `indices`, already on the device (see DeviceLoader below); it is the one way to read the big blocks, in batches. num_workers worker processes fetch and stack the examples on the CPU in parallel.
    # The device comes from this dataset (main() passes it to the constructor), so no consumer needs to know it.
    def loader(self, indices, batch_size, shuffle=False, generator=None, num_workers=0):
        # a Subset is a lazy view of the examples: it only holds the dataset and the indices (no data is copied or loaded), and reads an example from the dataset when it is fetched
        subset = torch.utils.data.Subset(self, indices)
        # "fork" as in the extraction: Python 3.14 defaults to "forkserver", which pickles the dataset to every worker; persistent workers survive between epochs
        loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, generator=generator,
            num_workers=num_workers, persistent_workers=num_workers > 0, multiprocessing_context="fork" if num_workers > 0 else None,
        )
        return DeviceLoader(loader, self._device)


# The one place that moves batches to the device: workers cannot use the GPU, so they stack the examples on the CPU and every batch is moved here, in the main process, with one copy per tensor
# (instead of per example, which is slow). A class with an __iter__ and not a generator, so that the loader can be iterated once per epoch.
class DeviceLoader:
    def __init__(self, loader, device):
        self._loader = loader
        self._device = device

    def __iter__(self):
        for batch in self._loader:
            t = time.perf_counter()  # TIMING
            # the batch is a nested dict (labels is a dict inside it), and tree_map applies the function to every tensor at any depth, so one call moves all of it
            batch = torch.utils._pytree.tree_map(lambda tensor: tensor.to(self._device), batch)
            torch.cuda.synchronize()  # TIMING
            TIMINGS["device move"] += time.perf_counter() - t  # TIMING
            yield batch



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


# seeds every random number generator and switches off nondeterministic GPU behaviour, so extractions and results are reproducible (same as the probe
# job in probe.py, plus numpy, cuDNN and TF32). bitwise equality still needs the same GPU, driver and library versions.
def set_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for deterministic cuBLAS matmuls; must be set before the first CUDA operation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # raises an error if an operation has no deterministic implementation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark mode picks kernels by timing, which can differ between runs
    torch.backends.cuda.matmul.allow_tf32 = False  # full float32 matmuls, no TF32 rounding
    torch.backends.cudnn.allow_tf32 = False


# Everything computes on ONE device (the GPU). main() picks it, moves the model and gives it to the datasets, which hold their tensors on it,
# so the methods never move data between devices.
def main():

    # ---- SETUP ----
    device = "cuda"
    # Usage (from the repo root): python -m interpret.interpret configs/main.yaml
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    icfg = cfg["interpret"]
    set_seeds(icfg["seed"])  # before the first CUDA operation
    model, train_cfg, checkpoint_step = load_model(icfg["checkpoint_path"], icfg["model_weights"])
    model = model.to(device)
    # The final Lambda puts each image on `device`, so the extractor never moves anything itself.
    transform = transforms.Compose([evaluation_preprocess(train_cfg), transforms.Lambda(lambda x: x.to(device))])
    print(f"Loaded checkpoint from {icfg['checkpoint_path']}")

    dataset_cfg = icfg["datasets"][icfg["dataset"]]
    dataset = load_dataset(icfg["dataset"], cfg["probe"]["dataset_roots"])

    # Extractions are cached in output_dir/data/<hash of settings> (layout in README.md); if a run died, info.json is
    # missing and loading crashes below: delete that directory and rerun.
    extract_cfg = icfg["extract"]
    settings = {
        "checkpoint_path": icfg["checkpoint_path"], "checkpoint_step": checkpoint_step, "model_weights": icfg["model_weights"],
        "dataset": icfg["dataset"], "dataset_config": dataset_cfg,
        "seed": icfg["seed"], "num_samples": extract_cfg["num_samples"], "layer_stride": extract_cfg["layer_stride"],
    }
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    data_dir = Path(icfg["output_dir"]) / "data" / settings_hash

    # ---- EXTRACTION ----
    if data_dir.exists():
        print(f"Reusing extraction in {data_dir}")
    else:
        print(f"Extracting features into {data_dir}...")
        extract_activations(model, dataset, transform, settings, extract_cfg, data_dir)
        (data_dir / "config.json").write_text(json.dumps(cfg, indent=2))  # the full config that produced these files (a reused extraction keeps its own)

    kept_layers = json.loads((data_dir / "info.json").read_text())["kept_layers"]  # in forward-pass order

    # ---- ANALYSIS ----
    # One results folder per extraction and set of methods, so other extractions or method subsets don't overwrite it.
    sorted_method_names = sorted(icfg["methods"])
    methods_label = "_".join(sorted_method_names)
    results_dir = Path(icfg["output_dir"]) / "results" / f"{settings_hash}_{methods_label}"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(json.dumps(cfg, indent=2))  # the full config of this run

    results = {}
    for name in icfg["methods"]:
        print(f"Running {name}...")
        # every method builds its own fresh datasets (one per kept layer) with exactly the blocks it needs: all data starts out on the CPU and moves to the device only when the method asks for it,
        # so methods never depend on each other. 
        if name == "effective_rank":
            datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = EffectiveRank(datasets)
        elif name == "cka":
            datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = CKA(datasets, icfg["seed"], icfg[name])
        elif name == "linear_probe":
            datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = LinearProbe(datasets, device, icfg["seed"], icfg[name])
        else:
            raise KeyError(f"Unknown method '{name}'; add it to the method dispatch in main() of interpret/interpret.py")
        method.run()
        results[name] = method.results

        # Each method saves the plots it defines into its own subfolder.
        plot_dir = results_dir / name
        plot_dir.mkdir(exist_ok=True)
        for plot_name, figure in method.plot().items():
            figure.savefig(plot_dir / f"{plot_name}.png", bbox_inches="tight")

    (results_dir / "results.json").write_text(json.dumps(results, indent=2))

    write_dashboard(results_dir, data_dir, cfg)
    print(f"dashboard saved to: {results_dir / 'dashboard.html'}")

if __name__ == "__main__":
    main()
