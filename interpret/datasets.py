# interpret/datasets.py: one class per analysis dataset, exposing `records` (image, metadata) for the sampler.
from pathlib import Path
from PIL import Image
import torch
from probe import DATASET_ROOTS, THUNDER_V2


# WILDS train + val tiles. Not probe.py's locked ClassificationDataset: it returns only (tensor, label), but we need the
# slide/node metadata. We still reuse its data sources, THUNDER_V2 and DATASET_ROOTS.
class WildsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root = DATASET_ROOTS["wilds"]
        records = []
        for split in ("train", "val"):
            spec = THUNDER_V2["classification"]["wilds"][split]
            for image_path, label in zip(spec["images"], spec["labels"]):
                parts = Path(image_path).parts
                slide = next(part for part in parts if part.startswith("patient_"))  # slide folder, e.g. patient_004_node_4
                node = slide.rsplit("_", 1)[-1]  # lymph-node index of the slide within its patient
                records.append((image_path, {"tumor": int(label), "slide": slide, "node": node}))
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, metadata = self.records[index]
        return Image.open(self.root / image_path).convert("RGB"), metadata


# Builds the dataset named by `identifier`.
def load_dataset(identifier):
    if identifier == "wilds":
        return WildsDataset()
    raise KeyError(f"Unknown dataset '{identifier}'; add it to load_dataset in interpret/datasets.py")
