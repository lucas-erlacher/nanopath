# interpret/effective_rank.py: per-layer effective rank of the centered mean-over-patch-tokens and CLS vectors.
# relevant papers: https://infoscience.epfl.ch/record/110188 (introduced the definition), https://arxiv.org/abs/2210.02885 (applied it to modern SSL models)

# NOTE:
# rank of embeddings collected for a certain layer across datapoints tells us "how many of its dimensions is this layer actually occupying".
# it will very rarely be the case that one embedding is an exact linear combination of other collected embeddings 
# (analogously it will very rarely be the case that a singular value of the embedding matrix will be exactly zero) 
# i.e. rank of a real world embedding matrix will almost always be full even if some dimensions are very close to being unused. 
# effective rank aims to be more robust: it normalizes singular values to a probability distribution and then computes it's exponentiated entropy 
# (how uncertain are we about a draw from this distribution) i.e. if one singular value dominates (even if the others are non-zero) entropy is close to 0 
# (ER=1, rank would have been full unless the other sing vals are exactly 0) - if sing vals are approx. uniformly distributed entropy is close to log(k) (ER=k). 

from matplotlib.figure import Figure
import torch


class EffectiveRank:
    def __init__(self, chunked_dataset, method_cfg):
        self.chunked_dataset = chunked_dataset
        self.method_cfg = method_cfg
        self.results = None  # {vector type: {layer: effective rank}}, set by run()

    def run(self):
        collected = {"patch_mean": {}, "cls": {}}

        # per layer, collect across all datapoints the CLS tokens and mean of patch tokens: collected[token type][layer] = list of num_datapoints many vectors.
        for i in range(len(self.chunked_dataset)):
            _, _, layers = self.chunked_dataset[i]

            for layer_name, groups in layers.items():
                if layer_name not in collected["patch_mean"]:
                    collected["patch_mean"][layer_name] = []
                    collected["cls"][layer_name] = []
                collected["patch_mean"][layer_name].append(groups["patch"].mean(dim=0))  # mean over patch tokens
                collected["cls"][layer_name].append(groups["cls"][0, :])  # the CLS group is (1, C), indexing out 0-th position makes it a plain C-vector

        # One effective rank value per (vector type, layer): results[vector type][layer]
        results = {}
        for vector_type, vectors_by_layer in collected.items():  # iterate over vector types (patch_mean, cls)
            results[vector_type] = {}

            for layer_name, vectors in vectors_by_layer.items():  # iterate over layers, first to last
                features = torch.stack(vectors)
                # NOTE:
                # center the embedding cloud - without it a strong offset of the whole cloud from the origin gives one very large singular value, 
                # which can dominate the normalized distribution and can push the entropy (and ER) down even if the variation inside the cloud is spread evenly.
                # after centering, the singular values only describe the variation within the cloud.
                features = features - features.mean(dim=0, keepdim=True)
                singular_values = torch.linalg.svdvals(features)
                # normalize to a probability distribution (the sum of the non-negative singular values is their L1 norm)
                probabilities = singular_values / singular_values.sum()
                entropy = torch.special.entr(probabilities).sum()
                results[vector_type][layer_name] = torch.exp(entropy).item()

        self.results = results

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
