# interpret/effective_rank.py: Track A method. Effective rank (exp of the entropy of the normalized singular
# values of the centered representations) per layer, for GAP and CLS sample vectors. Contract shared by all
# methods: Method(dataset, cfg).run() -> results dict; main() in interpret.py collects and saves the results.
import torch


class EffectiveRank:
    # dataset: ChunkedFeatureDataset yielding (image, label, layers); cfg: the `interpret.effective_rank` YAML block.
    def __init__(self, dataset, cfg):
        self.dataset, self.cfg = dataset, cfg

    def run(self):
        collected = {"gap": {}, "cls": {}}

        for i in range(len(self.dataset)):
            _, _, layers = self.dataset[i]

            for layer_name, groups in layers.items():
                if layer_name not in collected["gap"]:
                    collected["gap"][layer_name] = []
                    collected["cls"][layer_name] = []

                collected["gap"][layer_name].append(groups["patch"].mean(dim=0))
                # .clone() because groups["cls"][0] is a view that would keep the whole loaded chunk alive.
                collected["cls"][layer_name].append(groups["cls"][0].clone())

        results = {}
        for layer_name in collected["gap"]:
            layer_results = {}

            for representation_name, representations in collected.items():
                features = torch.stack(representations[layer_name])
                features = features - features.mean(dim=0, keepdim=True)
                sample_count, hidden_dim = features.shape
                if sample_count > hidden_dim:
                    covariance = (features.T @ features) / (sample_count - 1)
                    singular_values = torch.sqrt(torch.relu(torch.linalg.eigvalsh(covariance)))
                else:
                    singular_values = torch.linalg.svdvals(features)

                singular_values = singular_values[singular_values > self.cfg["singular_value_eps"]]
                if len(singular_values) == 0:
                    layer_results[f"{representation_name}_effective_rank"] = 0.0
                    continue

                probabilities = singular_values / singular_values.sum()
                entropy = -torch.sum(probabilities * torch.log(probabilities))
                layer_results[f"{representation_name}_effective_rank"] = torch.exp(entropy).item()

            results[layer_name] = layer_results

        return results
