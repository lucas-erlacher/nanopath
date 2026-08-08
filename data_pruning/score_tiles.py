import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.cluster import KMeans

from utils import assert_shape, print_device


def load_embeddings(embed_dir):
    embed_paths = sorted(Path(embed_dir).glob("*.embeddings.npz"))
    if not embed_paths:
        raise RuntimeError(f"no embedding shards found under {embed_dir}")

    paths, embeddings = [], []
    for shard_index, embed_path in enumerate(embed_paths, start=1):
        data = np.load(embed_path, allow_pickle=False)
        paths.extend(data["paths"].tolist())
        embeddings.append(data["embeddings"])
        print(f"loaded embedding shard {shard_index}/{len(embed_paths)}: {embed_path.name}", flush=True)

    return paths, np.concatenate(embeddings, axis=0)


def cluster_embeddings(embeddings, n_clusters, seed):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        verbose=1,
    )

    print(f"starting KMeans: rows={len(embeddings):,} clusters={n_clusters:,}", flush=True)
    labels = kmeans.fit_predict(embeddings)
    print(f"finished KMeans after {kmeans.n_iter_} iterations", flush=True)
    assigned_centroids = kmeans.cluster_centers_[labels]

    return labels, assigned_centroids


# vectorized over the number of embeddings dimension
def cosine_similarity(embeddings, centroids):
    num_embeddings = embeddings.shape[0]

    # * denotes embedding dimensionality (e.g. 384)
    assert_shape(embeddings, [num_embeddings, "*"], "embeddings")
    assert_shape(centroids, [num_embeddings, "*"], "centroids")

    matrix_elem_prod = embeddings * centroids  # elementwise product between 2 matrices
    assert_shape(matrix_elem_prod, [num_embeddings, "*"], "matrix_elem_prod") 

    dot_products = np.sum(matrix_elem_prod, axis=1) 
    assert_shape(dot_products, [num_embeddings], "dot products")

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    assert_shape(embedding_norms, [num_embeddings], "embedding norms")

    centroid_norms = np.linalg.norm(centroids, axis=1)
    assert_shape(centroid_norms, [num_embeddings], "centroid norms")
    
    normalizers = (embedding_norms * centroid_norms)  # element wise product
    assert_shape(normalizers, [num_embeddings], "normalizer")

    scores  = dot_products / normalizers  # element wise division
    assert_shape(scores, [num_embeddings], "scores")

    return scores 


def main():
    print_device("score_tiles")
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    embed_dir = Path(cfg["prune"]["embeddings_path"])
    out_path = Path(cfg["prune"]["scores_path"])

    started = time.monotonic()
    paths, embeddings = load_embeddings(embed_dir)
    print(f"loaded {len(paths):,} embeddings in {time.monotonic() - started:.1f}s", flush=True)

    cluster_labels, assigned_centroids = cluster_embeddings(
        embeddings,
        cfg["prune"]["num_clusters"],
        cfg["train"]["seed"],
    )

    similarities = cosine_similarity(embeddings, assigned_centroids)
    scores = 1.0 - similarities  # we want to encourage sampling of tiles that are dissimilar to their asigned centroids
    print("computed scores; writing parquet", flush=True)

    out = pa.table(
        {
            "path": paths,
            "score": scores.astype(np.float32),
            "cluster": cluster_labels.astype(np.int32),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path)
    print(f"wrote {out_path} in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()