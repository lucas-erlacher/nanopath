import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from clustering_utils import cluster_embeddings, load_embeddings
from utils import assert_shape, print_device


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

    paths, embeddings = load_embeddings(embed_dir, cfg)

    print("loaded embeddings")

    cluster_labels, cluster_centers = cluster_embeddings(
        embeddings,
        cfg["prune"]["num_clusters"],
        cfg["train"]["seed"],
    )
    assigned_centroids = cluster_centers[cluster_labels]

    print("clustered embeddings")

    similarities = cosine_similarity(embeddings, assigned_centroids)
    scores = 1.0 - similarities  # we want to encourage sampling of tiles that are dissimilar to their asigned centroids

    out = pa.table(
        {
            "path": paths,
            "score": scores.astype(np.float32),
            "cluster": cluster_labels.astype(np.int32),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path)

    print("done")


if __name__ == "__main__":
    main()