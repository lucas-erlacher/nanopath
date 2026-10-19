# interpret/datasets.py: one class per analysis dataset, exposing `records` (split, image, metadata) for the sampler.
from pathlib import Path
from PIL import Image
import torch
from probe import DATASET_ROOTS, THUNDER_V2


# WILDS train + val tiles. Not probe.py's locked ClassificationDataset: it returns only (tensor, label), but we need the
# patient/node metadata. We still reuse its data sources, THUNDER_V2 and DATASET_ROOTS.
class WildsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root = DATASET_ROOTS["wilds"]
        records = []
        for split in ("train", "val"):
            spec = THUNDER_V2["classification"]["wilds"][split]
            for image, target in zip(spec["images"], spec["labels"]):
                parts = Path(image).parts
                patient = next(part for part in parts if part.startswith("patient_"))
                records.append((split, image, {"tumor": int(target), "patient": patient, "node": patient.rsplit("_", 1)[-1]}))
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        split, image, metadata = self.records[index]
        return Image.open(self.root / image).convert("RGB"), {"split": split, "path": image, **metadata}


# Builds the dataset named by `identifier`.
def load_dataset(identifier):
    if identifier == "wilds":
        return WildsDataset()
    raise KeyError(f"Unknown dataset '{identifier}'; add it to load_dataset in interpret/datasets.py")
