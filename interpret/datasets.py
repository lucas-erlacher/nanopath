# interpret/datasets.py: one class per analysis dataset (image + metadata, plus `stratification_values` for the sampler) and load_dataset().
import json
from pathlib import Path
from PIL import Image
import torch


class WildsDataset(torch.utils.data.Dataset):
    # Camelyon17-WILDS tiles from the thunder benchmark manifest (train + val). Metadata fields available as
    # stratification dimensions: tumor (0/1), patient, node (hospital), split.
    def __init__(self, path, stratification_dimension):
        self.root = Path(path)
        manifest = json.loads((Path(__file__).resolve().parent.parent / "benchmarking" / "thunder_v2.json").read_text())
        self.records = []
        for split in ("train", "val"):
            spec = manifest["classification"]["wilds"][split]
            for image, target in zip(spec["images"], spec["labels"]):
                patient = next(part for part in Path(image).parts if part.startswith("patient_"))
                self.records.append((image, {"split": split, "tumor": int(target), "patient": patient, "node": patient.rsplit("_", 1)[-1]}))
        self.stratification_values = [metadata[stratification_dimension] for _, metadata in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, metadata = self.records[index]
        return Image.open(self.root / image).convert("RGB"), {"path": image, **metadata}


# Builds the dataset named by `identifier` from its config block ({path, stratification_dimension}).
def load_dataset(identifier, dataset_cfg):
    if identifier == "wilds":
        return WildsDataset(dataset_cfg["path"], dataset_cfg["stratification_dimension"])
    raise KeyError(f"Unknown dataset '{identifier}'; add it to load_dataset in interpret/datasets.py")
