import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.cluster import MiniBatchKMeans


def load_embeddings(embed_path):
    data = np.load(embed_path, allow_pickle=False)
    return data["paths"].tolist(), data["embeddings"]


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

    embed_path = Path(pembed_batche_cfg["output_path"]).with_suffix(".embeds.npz")
    paths, embeddings = load_embeddings(embed_path)

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