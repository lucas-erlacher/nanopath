# Build hierarchical tile weights as a drop-in alternative to flat tile scores.

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from clustering_utils import cluster_embeddings, load_embeddings


################    CLUSTERING HIERARCHY    ################


def build_hierarchy(embeddings, cluster_counts, seed):
    if len(cluster_counts) < 2 or cluster_counts[-1] != 1 or any(left <= right for left, right in zip(cluster_counts, cluster_counts[1:])):
        raise RuntimeError("hierarchical_cluster_counts must have at least two strictly descending counts and end with 1")

    return _build_mappings(embeddings, cluster_counts, seed)


def _build_mappings(datapoints, cluster_counts, seed):
    if not cluster_counts:
        return []

    # cluster_ids is a 1D array containing for each datapoint the index of the centroid it was assigned to
    cluster_ids, centroids = cluster_embeddings(datapoints, cluster_counts[0], seed)
    return [cluster_ids] + _build_mappings(centroids, cluster_counts[1:], seed)


################    WEIGHT COMPUTATION    ################


def hierarchical_weights(mappings):
    weights = np.zeros(len(mappings[0]), dtype=np.float64)

    root_centroid_id = int(mappings[-1][0])
    probability_mass = 1.0

    split_budget(probability_mass, root_centroid_id, mappings, weights)

    # numerical sanity check
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("hierarchical weights must be finite, non-negative, and sum to one")

    return weights


def split_budget(budget, parent_centroid_id, mappings, weights):
    # load mapping of level we are currently working on
    child_to_parent_ids = mappings[-1]  # 1D array containing for each datapoint the index of the centroid it was assigned to

    child_is_in_cluster = child_to_parent_ids == parent_centroid_id  # boolean mask for every child if child is in cluster spanned by parent_centroid_id
    indices = np.arange(len(child_is_in_cluster))  # range of indices in child_to_parent_ids
    child_centroid_ids = indices[child_is_in_cluster]  # indices of children that are in cluster spanned by parent_centroid_id

    # find proportion of budget that should flow into each child
    cluster_size = len(child_centroid_ids)
    child_budget = budget / cluster_size

    if len(mappings) == 1:
        # end of recursion: these indices are data points
        weights[child_centroid_ids] = child_budget
    else:
        # recurse onto children, passing them the new budget
        for child_centroid_id in child_centroid_ids:
            split_budget(child_budget, child_centroid_id, mappings[:-1], weights)


################    MAIN    ################


def sanity_check():
    # small end-to-end check: five data points, three centroids, and one root centroid.
    test_embeddings = np.array([[0, 0], [0, 1], [10, 10], [10, 11], [20, 20]], dtype=np.float32)
    test_mappings = build_hierarchy(test_embeddings, [3, 1], seed=0)
    test_weights = hierarchical_weights(test_mappings)
    expected_weights = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3])
    assert len(test_mappings) == 2
    assert len(test_mappings[0]) == len(test_embeddings)
    assert np.allclose(test_weights, expected_weights)
    print(f"sanity check weights: {test_weights}")


def main():
    sanity_check()
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    embeddings_dir = Path(cfg["prune"]["embeddings_path"])
    paths, embeddings = load_embeddings(embeddings_dir, cfg)
    cluster_counts = [int(count) for count in cfg["prune"]["hierarchical_cluster_counts"]]

    mappings = build_hierarchy(
        embeddings,
        cluster_counts,
        int(cfg["train"]["seed"]),
    )

    weights = hierarchical_weights(mappings)

    # save the computed weights
    out_path = Path(cfg["prune"]["scores_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "path": paths,
            "score": weights.astype(np.float32),
            "cluster": mappings[0].astype(np.int32),
        }),
        out_path,
    )


if __name__ == "__main__":
    main()