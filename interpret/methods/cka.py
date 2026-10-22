# interpret/methods/cka.py: linear CKA between all pairs of layers, for the mean-over-patch-tokens and CLS vectors (the steps are in README.md).
# Follows the table in README.md (see "step n" comments) and the paper authors' demo code (google-research/representation_similarity/Demo.ipynb).

from matplotlib.figure import Figure
import torch
from tqdm import tqdm



class CKA:
    def __init__(self, pooled_vectors, seed, method_cfg):
        self.pooled_vectors = pooled_vectors  # {vector type: {layer: (n, C) matrix}}, see pool_vectors in interpret.py
        self.seed = seed
        self.method_cfg = method_cfg  # max_samples: cap on the number of examples
        self.results = None  # {vector type: {layer a: {layer b: CKA}}}, set by run()

    def run(self):
        results = {}

        for vector_type, features_by_layer in self.pooled_vectors.items():  # iterate over vector types (patch_mean, cls)
            flattened_tables = {}  # one n x n table per layer, flattened to a vector, all held at once: max_samples caps the RAM
            squared_lengths = {}

            # compute the similarity structure tables for each layer (and perform a precomputation for step 6)
            for layer_name, features in tqdm(features_by_layer.items(), desc=f"build similarity structure tables ({vector_type})"):  # iterate over layers, first to last
                # the n x n tables may need more RAM than the data allows, so keep a random subset of at most max_samples examples (label-free, so not exactly stratified). 
                # the same seed gives the same permutation for the same n, so every layer keeps the same examples.
                # a random subset of k examples from the stratified extraction shifts a class proportion p by at most about sqrt(p(1-p)/k) (an estimate: 0.7 percentage points at k = 5,000
                # for two balanced classes), so the subset stays close to stratified, especially at large max_samples.
                permutation = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(self.seed))
                features = features[permutation[:self.method_cfg["max_samples"]], :]
                num_examples = features.shape[0]

                # step 4-5 (center features: HX)
                features = features - features.mean(dim=0, keepdim=True)  # subtract the mean feature from every feature (centering)

                # step 2 (compute similarity structure table (of the centered features, so it is the centered table HKH of step 5))
                table = features @ features.T  # similarity structure matrix of layer

                # step 3 (flatten)
                flattened_table = table.view(num_examples ** 2)  # a view and not reshape, to avoid copy of the table
                flattened_tables[layer_name] = flattened_table

                # step 6 (precomputation for normalization step)
                squared_lengths[layer_name] = torch.dot(flattened_table, flattened_table)  # inner product of a flattened table with itself i.e. sum of squared entries

            # compare the similarity structure tables between layers
            results[vector_type] = {}
            for layer_a in tqdm(flattened_tables, desc=f"compare similarity structure tables ({vector_type})"):
                results[vector_type][layer_a] = {}
                for layer_b in flattened_tables:
                    # step 3 (unnormalized comparison of two layer-tables)
                    inner_product = torch.dot(flattened_tables[layer_a], flattened_tables[layer_b])

                    # step 6 (normalization)
                    normalizer = (squared_lengths[layer_a] * squared_lengths[layer_b]).sqrt()  # product of the two lengths (Frobenius norms)
                    cka = inner_product / normalizer

                    results[vector_type][layer_a][layer_b] = cka.item()

        self.results = results

    # One heatmap per vector type, layers in forward-pass order on both axes. Returns {file name: figure}; main() saves them.
    def plot(self):
        figures = {}
        for vector_type, title in (("patch_mean", "mean over the patch tokens"), ("cls", "CLS token")):
            layer_names = list(self.results[vector_type])
            matrix = []
            for layer_a in layer_names:
                row = []
                for layer_b in layer_names:
                    row.append(self.results[vector_type][layer_a][layer_b])
                matrix.append(row)
            figure = Figure()
            axis = figure.subplots()
            image = axis.imshow(matrix, origin="lower", vmin=0, vmax=1)
            axis.set_xticks(range(len(layer_names)), layer_names, rotation=90)
            axis.set_yticks(range(len(layer_names)), layer_names)
            axis.set_xlabel("layer (increasing depth)")
            axis.set_ylabel("layer (increasing depth)")
            axis.set_title(f"CKA of the {title} between layers")
            figure.colorbar(image, ax=axis)
            figures[f"{vector_type}_cka"] = figure
        return figures