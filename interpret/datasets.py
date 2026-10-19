# interpret/datasets.py: one class per analysis dataset, exposing `records` (image, metadata) for the sampler.
import io
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch


# PathoROB Camelyon tiles, loaded as in probe.py's PathoROB probe. Reimplemented because that code is inline in a larger
# function and returns only the transformed image; we also expose the metadata: slide_id, biological_class, medical_center.
class PathoRobCamelyonDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        table = pa.concat_tables([pq.read_table(f) for f in sorted((root / "camelyon").glob("data/*.parquet"))])
        images = [r["bytes"] for r in table.column("image").to_pylist()]
        metadata = table.select(["slide_id", "biological_class", "medical_center"]).to_pylist()
        self.records = list(zip(images, metadata))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_bytes, metadata = self.records[index]
        return Image.open(io.BytesIO(image_bytes)).convert("RGB"), metadata


# Builds the dataset named by `identifier`; `dataset_roots` is probe.dataset_roots from the config (name -> folder).
def load_dataset(identifier, dataset_roots):
    if identifier == "pathorob_camelyon":
        return PathoRobCamelyonDataset(Path(dataset_roots["pathorob"]))
    raise KeyError(f"Unknown dataset '{identifier}'; add it to load_dataset in interpret/datasets.py")
