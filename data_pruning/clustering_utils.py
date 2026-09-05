# Shared embedding and k-means preprocessing helpers for flat and hierarchical clustering.
# Keep shared preprocessing behavior identical across both offline paths.

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def load_embeddings(embed_dir, cfg):
    embed_paths = sorted((Path(embed_dir) / cfg["prune"]["embedding_model"]).glob("*.embeddings.npz"))
    if not embed_paths:
        raise RuntimeError(f"no embedding shards found under {embed_dir}")
    paths, embeddings = [], []
    for embed_path in embed_paths:
        data = np.load(embed_path, allow_pickle=False)
        paths.extend(data["paths"].tolist())
        embeddings.append(data["embeddings"])
    if len(paths) != sum(batch.shape[0] for batch in embeddings) or len(set(paths)) != len(paths):
        raise RuntimeError("embedding paths must be unique and aligned with embeddings")
    return paths, np.concatenate(embeddings, axis=0)


def cluster_embeddings(embeddings, n_clusters, seed):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        verbose=1,
    )
    return kmeans.fit_predict(embeddings), kmeans.cluster_centers_
