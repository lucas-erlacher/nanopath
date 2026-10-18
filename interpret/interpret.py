import os
import math
import json
import random
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, List, Any
from pathlib import Path
from PIL import Image

from model import load_model
from probe import worker_probe_transforms


# -------------------------------------------------------------------------------------------------
# INTERPRETABILITY ANALYSIS
# -------------------------------------------------------------------------------------------------

def run_effective_rank_analysis(dataset: torch.utils.data.Dataset) -> dict:
    collected = {"gap": {}, "cls": {}}

    for i in range(len(dataset)):
        _, _, layers = dataset[i]

        for layer_name, groups in layers.items():
            if layer_name not in collected["gap"]:
                collected["gap"][layer_name] = []
                collected["cls"][layer_name] = []

            # .clone() breaks the memory reference, preventing the chunk data tensor from causing an OOM leak in RAM.
            gap_repr = groups["patch"].mean(dim=0).clone()
            collected["gap"][layer_name].append(gap_repr)

            if "cls" in groups:
                # .clone() breaks the memory reference, preventing the chunk data tensor from causing an OOM leak in RAM.
                cls_repr = groups["cls"][0].clone()
                collected["cls"][layer_name].append(cls_repr)

    results = {}
    for layer_name in collected["gap"]:
        layer_results = {}

        for representation_name, representations in collected.items():
            if not representations[layer_name]:
                layer_results[f"{representation_name}_effective_rank"] = None
                continue

            features = torch.stack(representations[layer_name]).float()
            features = features - features.mean(dim=0, keepdim=True)
            sample_count, hidden_dim = features.shape
            if sample_count > hidden_dim:
                covariance = (features.T @ features) / (sample_count - 1)
                singular_values = torch.sqrt(torch.relu(torch.linalg.eigvalsh(covariance)))
            else:
                singular_values = torch.linalg.svdvals(features)

            singular_values = singular_values[singular_values > 1e-9]
            if len(singular_values) == 0:
                layer_results[f"{representation_name}_effective_rank"] = 0.0
                continue

            probabilities = singular_values / singular_values.sum()
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            layer_results[f"{representation_name}_effective_rank"] = torch.exp(entropy).item()

        results[layer_name] = layer_results

    return results




# -------------------------------------------------------------------------------------------------
# REPRESENTATION EXTRACTION
# -------------------------------------------------------------------------------------------------

def assert_hook_interface(
    storage_dict: Dict[str, Dict[str, torch.Tensor]],
    layer_names: List[str],
    batch_size: int,
    global_state: Dict[str, Any],
):
    assert len(layer_names) == len(set(layer_names)), "Duplicate layer keys in returned list."

    for layer in layer_names:
        assert layer in storage_dict, f"Layer '{layer}' missing from hook output."
        group_dict = storage_dict[layer]
        assert "patch" in group_dict, f"Layer '{layer}' missing 'patch' group."

        current_keys = set(group_dict.keys())
        if global_state["expected_keys"] is None:
            global_state["expected_keys"] = current_keys
        expected_keys = global_state["expected_keys"]
        assert current_keys == expected_keys, f"Layer '{layer}' keys {current_keys} mismatch global {expected_keys}."

        layer_expected_dim = None
        for group_name, tensor in group_dict.items():
            assert tensor.ndim == 3, f"Layer '{layer}' group '{group_name}' not 3D."
            batch, tokens, hidden_dim = tensor.shape
            assert batch == batch_size, f"Layer '{layer}' group '{group_name}' batch {batch} != {batch_size}."

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

def _process_single_sample_features(
    storage_dict: Dict[str, Dict[str, torch.Tensor]],
    kept_layer_names: List[str],
    batch_idx: int,
    on_nonfinite: str
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Helper to flatten the extraction loop and handle precision/device casting."""
    sample_layers = {}
    for layer in kept_layer_names:
        sample_layers[layer] = {}
        for group_name, tensor in storage_dict[layer].items():
            item_tensor = tensor[batch_idx]
            formatted_tensor = item_tensor.to(dtype=torch.float16, device="cpu").contiguous()

            if not torch.isfinite(formatted_tensor).all():
                msg = f"Non-finite float16 values in {layer}:{group_name}."
                if on_nonfinite == 'raise':
                    raise ValueError(msg)
                print(f"WARNING: {msg}")

            sample_layers[layer][group_name] = formatted_tensor

    return sample_layers

def extract_features_to_chunks(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    output_dir: str,
    transform,
    chunk_size: int = 500,
    num_samples: int = 1000,
    layer_stride: int = 1,
    seed: int = 42,
    on_nonfinite: str = 'raise'
):
    random.seed(seed)
    num_available = len(dataset)
    sampled_indices = random.sample(range(num_available), min(num_samples, num_available))
    subset = torch.utils.data.Subset(dataset, sampled_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

    os.makedirs(output_dir, exist_ok=True)

    storage_dict = {}
    layer_names = model.register_interpretability_hooks(storage_dict)
    kept_layer_names = layer_names[::-layer_stride][::-1]

    # State tracking to strictly enforce consistency across ALL layers and ALL batches
    interface_state = {"expected_keys": None}

    chunk_idx = 0
    global_idx = 0
    current_chunk_data = []
    sample_mapping = []

    model.eval()
    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_size = len(batch_images)
            processed_images = torch.stack([transform(image) for image in batch_images])

            # Flush hook buffer natively in-place
            storage_dict.clear()
            _ = model(processed_images)

            # Validate rigorous contract (before stride)
            assert_hook_interface(storage_dict, layer_names, batch_size, interface_state)

            for b in range(batch_size):
                sample_features = _process_single_sample_features(
                    storage_dict, kept_layer_names, b, on_nonfinite
                )

                label_val = batch_labels[b]
                if torch.is_tensor(label_val):
                    label_val = label_val.cpu()

                datapoint = {
                    "image": processed_images[b].cpu(),
                    "label": label_val,
                    "layers": sample_features
                }
                current_chunk_data.append(datapoint)

                chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx}.pt")
                sample_mapping.append({
                    "original_idx": sampled_indices[global_idx],
                    "chunk_path": chunk_path,
                    "index_in_chunk": len(current_chunk_data) - 1
                })

                global_idx += 1

                if len(current_chunk_data) >= chunk_size:
                    torch.save(current_chunk_data, chunk_path)
                    chunk_idx += 1
                    current_chunk_data = []

        # Flush remainder
        if current_chunk_data:
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx}.pt")
            torch.save(current_chunk_data, chunk_path)

    paths_data = {
        "seed": seed,
        "layer_stride": layer_stride,
        "kept_layers": kept_layer_names,
        "sampled_dataset_indices": sampled_indices,
        "sample_mapping": sample_mapping
    }
    with open(os.path.join(output_dir, "paths.json"), 'w') as f:
        json.dump(paths_data, f)




# -------------------------------------------------------------------------------------------------
# DATASET CLASSES
# -------------------------------------------------------------------------------------------------

class WildsDataset(torch.utils.data.Dataset):
    def __init__(self, root, records):
        self.root = root
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        split, image, metadata = self.records[index]
        return Image.open(self.root / image).convert("RGB"), {"split": split, "path": image, **metadata}

class ChunkedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, paths_file: str):
        with open(paths_file, 'r') as f:
            metadata = json.load(f)

        self.sample_mapping = metadata["sample_mapping"]
        self.current_chunk_path = None
        self.current_chunk_data = None

    def __len__(self):
        return len(self.sample_mapping)

    def __getitem__(self, idx):
        mapping = self.sample_mapping[idx]
        chunk_path = mapping["chunk_path"]

        # Zero-overhead caching optimized for single-process sequential iteration
        if self.current_chunk_path != chunk_path:
            self.current_chunk_data = torch.load(chunk_path, weights_only=False)
            self.current_chunk_path = chunk_path

        datapoint = self.current_chunk_data[mapping["index_in_chunk"]]
        return datapoint["image"], datapoint["label"], datapoint["layers"]




# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

def evaluation_preprocess(cfg):
    transform, _ = worker_probe_transforms(cfg)
    # Normalization is part of probe.py's preprocessing, but it is applied at probe model-call
    # sites rather than inside worker_probe_transforms(). Since probe.py must not be edited, we
    # reimplement it here. Regularly compare the pipelines to prevent drift (e.g. with an agent).
    mean = torch.tensor(cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["data"]["std"]).view(3, 1, 1)
    return transforms.Compose([transform, transforms.Lambda(lambda x: (x - mean) / std)])

def load_dataset(cfg) -> torch.utils.data.Dataset:
    root = Path(cfg["probe"]["dataset_roots"]["wilds"])
    manifest = json.loads((Path(__file__).resolve().parent.parent / "benchmarking" / "thunder_v2.json").read_text())
    records = []
    for split in ("train", "val"):
        spec = manifest["classification"]["wilds"][split]
        for image, target in zip(spec["images"], spec["labels"]):
            parts = Path(image).parts
            patient = next(part for part in parts if part.startswith("patient_"))
            records.append((split, image, {"tumor": int(target), "patient": patient, "node": patient.rsplit("_", 1)[-1]}))

    return WildsDataset(root, records)


def main(checkpoint_path: str = None, num_samples: int = 1000):
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_model(checkpoint_path, torch.device("cuda"))
    transform = evaluation_preprocess(checkpoint["config"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    dataset = load_dataset(checkpoint["config"])
    output_dir = "./interp_chunks"
    print("Extracting features...")
    extract_features_to_chunks(model, dataset, output_dir, transform, num_samples=num_samples)

    paths_file = os.path.join(output_dir, "paths.json")
    print("Computing Effective Rank...")
    dataset = ChunkedFeatureDataset(paths_file)
    er_results = run_effective_rank_analysis(dataset)

    for layer, metrics in er_results.items():
        cls_str = f", CLS ER: {metrics['cls_effective_rank']:.2f}" if metrics['cls_effective_rank'] else ""
        print(f"[{layer}] GAP ER: {metrics['gap_effective_rank']:.2f}{cls_str}")

if __name__ == "__main__":
    main()