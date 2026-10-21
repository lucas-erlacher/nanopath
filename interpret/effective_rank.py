# interpret/effective_rank.py: per-layer effective rank of the centered mean-over-patch-tokens and CLS vectors.

from matplotlib.figure import Figure
import torch
from tqdm import tqdm


class EffectiveRank:
    def __init__(self, chunked_dataset, method_cfg):
        self.chunked_dataset = chunked_dataset
        self.method_cfg = method_cfg
        self.results = None  # {vector type: {layer: effective rank}}, set by run()

    def run(self):
        collected = {"patch_mean": {}, "cls": {}}

        # per layer, collect across all datapoints the CLS tokens and mean of patch tokens: collected[token type][layer] = list of num_datapoints many vectors.
        for i in tqdm(range(len(self.chunked_dataset)), desc="Collecting"):
            _, _, layers = self.chunked_dataset[i]

            for layer_name, groups in layers.items():
                if layer_name not in collected["patch_mean"]:
                    collected["patch_mean"][layer_name] = []
                    collected["cls"][layer_name] = []
                collected["patch_mean"][layer_name].append(groups["patch"].mean(dim=0))  # mean over patch tokens
                collected["cls"][layer_name].append(groups["cls"][0, :])  # the CLS group is (1, C), indexing out 0-th position makes it a plain C-vector

        print(f"Collected {len(self.chunked_dataset)} samples for {len(collected['cls'])} layers, computing effective ranks")

        # One effective rank value per (vector type, layer): results[vector type][layer]
        results = {}
        for vector_type, vectors_by_layer in collected.items():  # iterate over vector types (patch_mean, cls)
            results[vector_type] = {}

            for layer_name, vectors in tqdm(vectors_by_layer.items(), desc=f"Effective rank ({vector_type})"):  # iterate over layers, first to last
                features = torch.stack(vectors)
                # center the embedding cloud (PCA-style) so the singular values describe the variation within it, not its offset from the origin
                features = features - features.mean(dim=0, keepdim=True)
                singular_values = torch.linalg.svdvals(features)
                # normalize to a probability distribution (the sum of the non-negative singular values is their L1 norm)
                probabilities = singular_values / singular_values.sum()
                entropy = torch.special.entr(probabilities).sum()
                results[vector_type][layer_name] = torch.exp(entropy).item()

        self.results = results
        print("Effective rank done")

    # One line plot per representation: effective rank per layer. self.results is in forward-pass order, so the x axis
    # runs from the first to the last layer. Returns {file name: figure}; main() saves them.
    def plot(self):
        figures = {}
        for vector_type, title in (("patch_mean", "mean over the patch tokens"), ("cls", "CLS token")):
            layer_names = list(self.results[vector_type])
            ranks = []
            for layer_name in layer_names:
                ranks.append(self.results[vector_type][layer_name])
            figure = Figure()
            axis = figure.subplots()
            axis.plot(layer_names, ranks, marker="o")
            axis.set_xlabel("layer (increasing depth)")
            axis.set_ylabel("effective rank")
            axis.set_title(f"Effective rank of the {title} per layer")
            figures[f"{vector_type}_effective_rank"] = figure
        return figures
