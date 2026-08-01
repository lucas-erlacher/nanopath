import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torchvision.transforms import v2

from model import DinoV2ViT, load_dinov2_pretrained

EMBED_BATCH_SIZE = 256


@torch.no_grad()
def embed_tiles(dataset_dir, mean, std, device):
    backbone = load_dinov2_pretrained(DinoV2ViT(variant="dinov2_vits14_reg")).to(device).eval()

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    paths, batch, embeddings = [], [], []

    for shard_path in sorted(Path(dataset_dir).glob("shard-*.parquet")):
        table = pq.read_table(str(shard_path), columns=["path", "jpeg"])
        for path, jpeg in zip(table["path"].to_pylist(), table["jpeg"].to_pylist()):
            batch.append(to_tensor(Image.open(io.BytesIO(jpeg)).convert("RGB")))
            paths.append(path)

            if len(batch) == EMBED_BATCH_SIZE:
                embeddings.append(embed_batch(batch, device, autocast, backbone, mean_t, std_t))
                batch = []

    if batch:
        embeddings.append(embed_batch(batch, device, autocast, backbone, mean_t, std_t))

    return paths, np.concatenate(embeddings, axis=0)

def embed_batch(batch, device, autocast, backbone, mean_t, std_t):
    x = torch.stack(batch).to(device)
    with autocast:
        return (
            backbone.probe_features((x - mean_t) / std_t)
            .float()
            .cpu()
            .numpy()
        )


def cluster_embeddings(embeddings, n_clusters, seed):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
    )

    labels = kmeans.fit_predict(embeddings)
    assigned_centroids = kmeans.cluster_centers_[labels]

    return labels, assigned_centroids


def cosine_distance(embeddings, references):
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    references = references / np.linalg.norm(
        references, axis=1, keepdims=True
    )

    return 1.0 - np.sum(embeddings * references, axis=1)


def main():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    pembed_batche_cfg = cfg["pembed_batche"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, embeddings = embed_tiles(
        cfg["data"]["dataset_dir"],
        cfg["data"]["mean"],
        cfg["data"]["std"],
        device,
    )

    cluster_labels, assigned_centroids = cluster_embeddings(
        embeddings,
        pembed_batche_cfg["n_clusters"],
        cfg["train"]["seed"],
    )

    scores = cosine_distance(embeddings, assigned_centroids)

    out = pa.table(
        {
            "path": paths,
            "score": scores.astype(np.float32),
            "cluster": cluster_labels.astype(np.int32),
        }
    )

    pq.write_table(out, pembed_batche_cfg["output_path"])


if __name__ == "__main__":
    main()