# interpret/datasets.py: one class per supported analysis dataset, plus load_dataset() which main() calls.
# Contract: Dataset(path, stratification_dimension) is a torch Dataset whose __getitem__ returns (PIL RGB image,
# metadata dict) and which exposes `stratification_values`, one hashable value per sample taken from the metadata
# field named by the stratification dimension. The sampler in interpret.py draws equally from every value.
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
                self.records.append((split, image, {"tumor": int(target), "patient": patient, "node": patient.rsplit("_", 1)[-1]}))
        self.stratification_values = [{"split": split, **meta}[stratification_dimension] for split, _, meta in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        split, image, metadata = self.records[index]
        return Image.open(self.root / image).convert("RGB"), {"split": split, "path": image, **metadata}


# Builds the dataset named by `identifier` from its config block ({path, stratification_dimension}).
def load_dataset(identifier, cfg):
    if identifier == "wilds":
        return WildsDataset(cfg["path"], cfg["stratification_dimension"])
