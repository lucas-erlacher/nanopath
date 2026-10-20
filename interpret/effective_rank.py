# interpret/effective_rank.py: per-layer effective rank of the centered GAP and CLS vectors.
from collections import defaultdict
import torch


class EffectiveRank:
    def __init__(self, chunked_dataset, method_cfg):
        self.chunked_dataset = chunked_dataset
        self.method_cfg = method_cfg

    def run(self):
        collected = {"gap": defaultdict(list), "cls": defaultdict(list)}

        for i in range(len(self.chunked_dataset)):
            _, _, layers = self.chunked_dataset[i]

            for layer_name, groups in layers.items():
                collected["gap"][layer_name].append(groups["patch"].mean(dim=0))
                # clone: a view would keep the whole loaded chunk alive
                collected["cls"][layer_name].append(groups["cls"][0, :].clone())

        results = {}
        for layer_name in collected["gap"]:
            layer_results = {}

            for representation_name, representations in collected.items():
                features = torch.stack(representations[layer_name])
                features = features - features.mean(dim=0, keepdim=True)
                singular_values = torch.linalg.svdvals(features)

                singular_values = singular_values[singular_values > self.method_cfg["singular_value_eps"]]
                if len(singular_values) == 0:
                    layer_results[f"{representation_name}_effective_rank"] = 0.0
                    continue

                probabilities = singular_values / singular_values.sum()
                entropy = -torch.sum(probabilities * torch.log(probabilities))
                layer_results[f"{representation_name}_effective_rank"] = torch.exp(entropy).item()

            results[layer_name] = layer_results

        return results
