# Offline tile-sampling feature helpers. Disabled sampling returns a uniform baseline without score-file access.
# Score loading lives in sampling_metadata.py so this module only computes offline weights.

import torch


class OfflineSampler:
    def __init__(self, prune_cfg, scores, num_paths, is_train):
        if not is_train:
            self.weights = None
            return

        prune_enabled = bool(prune_cfg["enabled"])
        uniform_weights = torch.full((num_paths,), 1.0 / num_paths, dtype=torch.float64)
        if not prune_enabled:
            self.weights = uniform_weights
            return

        # finalize score-sampling distribution only when explicitly enabled.
        scores = torch.as_tensor(scores, dtype=torch.float64)
        cut_score = prune_cfg["cut_score"]
        removal_mask = torch.zeros_like(scores, dtype=torch.bool) if cut_score is None else scores > float(cut_score)
        scores[removal_mask] = 0
        score_distribution = scores / scores.sum()
        self.weights = torch.lerp(uniform_weights, score_distribution, float(prune_cfg["sampling_intensity"]))
        self.weights[removal_mask] = 0
