# Static tile metadata shared by offline and online sampling.
# The dataset builds this once so parquet scores and cluster IDs stay aligned.

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class SamplingMetadata:
    def __init__(self, scores, cluster_ids):
        self.scores = scores
        self.cluster_ids = cluster_ids


def load_sampling_metadata(prune_cfg, paths, need_scores, need_cluster_ids, is_train):
    cluster_ids = np.zeros(len(paths), dtype=np.int32)
    if not is_train or not need_scores and not need_cluster_ids:
        return SamplingMetadata(None, cluster_ids)

    scores_path = Path(prune_cfg["scores_path"])
    columns = ["path"]
    if need_scores:
        columns.append("score")
    if need_cluster_ids:
        columns.append("cluster")
    score_table = pq.read_table(str(scores_path), columns=columns)
    table_paths = score_table["path"].to_pylist()
    scores = None
    if need_scores:
        path_to_score = dict(zip(table_paths, score_table["score"].to_pylist()))
        scores = np.asarray([path_to_score[path] for path in paths], dtype=np.float64)
    if need_cluster_ids:
        path_to_cluster = dict(zip(table_paths, score_table["cluster"].to_pylist()))
        cluster_ids = np.asarray([path_to_cluster[path] for path in paths], dtype=np.int32)
    return SamplingMetadata(scores, cluster_ids)
