# interpret/methods/linear_probe.py: linear probes on the [CLS, patch mean] embedding of every layer, for the targets in the config (the spec is in README.md).

# DATA LEAKAGE PREVENTION (the primary objective, it always wins), in two places marked in the code:
#  - in the split: StratifiedGroupKFold puts all tiles with the same leakage_group value (a slide, in our config) on one side of every split, so the validation fold never contains a group the probe has trained on.
#  - in the standardization: the mean and std come from the training fold only, so no statistic of the validation fold flows into the training.
#
# CLASS BALANCING (the secondary objective), in two places marked in the code. They act on different things and do not stand in for each other:
#  - in the split (best effort): within the group-splitting constraint (that one is primary as leakage prevention is primary) it also tries on a best effort basis to give every fold the same class mix, i.e. every fold looks like the whole dataset, however skewed that is. 
#    it does not remove any skew. the class mix is that of the target being predicted: the folds are computed separately for every target and stratified on its labels (shared by all layers and grid points of that target, so the layers are compared on the same folds). 
#  - in the training loss: correct the skew the optimizer sees in training data, counts come from the training fold only.
#  why no loss balancing in evaluation: the weights correct what the probe learns i.e. balancing is central to training - balancing is not essential for validation loss computation i.e. we prefer to have an undistorted performance picture in evaluation.

import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

METRICS = ("balanced_accuracy", "macro_f1", "macro_auroc")  # the first one selects the best grid point


class LinearProbe:
    def __init__(self, layerwise_pooled_embeddings, seed, method_cfg):
        self._pooled = layerwise_pooled_embeddings  # see LayerwisePooledEmbeddings in interpret.py
        self.seed = seed
        self._cfg = method_cfg  
        self.results = None  

        self.layer_names = list(self._pooled.records["cls"])

        # the metadata of every example, only used here to build the folds and the briefing (the inputs and the labels of a fold are read from its records, see prepare_examples)
        metadatas = []
        group_ids = []
        for record in self._pooled.records["cls"][self.layer_names[0]]:  # going over embeddings of layer 1 covers all datapoints i.e. all metadata
            metadatas.append(record["metadata"])
            group_ids.append(record["metadata"][self._cfg["leakage_group"]])

        # pre-compute folds (once per target)
        # DATA LEAKAGE PREVENTION
        # CLASS BALANCING
        self.folds = {}
        for target in self._cfg["targets"]:
            labels = [metadata[target] for metadata in metadatas]
            splitter = StratifiedGroupKFold(n_splits=self._cfg["folds"], shuffle=True, random_state=seed)
            self.folds[target] = []
            for train_indices, val_indices in splitter.split(list(range(self._pooled.num_examples)), labels, groups=group_ids):
                self.folds[target].append((train_indices.tolist(), val_indices.tolist()))

        # lookup table mapping class names (strings etc) to integers.
        # the class set of every target is fixed once (sorted), so a class always gets the same integer label
        self.class_index = {}
        for target in self._cfg["targets"]:
            class_values = set()
            # collect possible vales
            for metadata in metadatas:
                class_values.add(metadata[target])
            # compute mapping from value to id (= the integer index)
            self.class_index[target] = {}
            for index, class_value in enumerate(sorted(class_values)):
                self.class_index[target][class_value] = index

        self.briefing = self.print_briefing(metadatas, group_ids)

    # -------------------------------------------------------------------------------------------------
    # TRAINING + EVALUATION
    # -------------------------------------------------------------------------------------------------

    def _setting_name(self, lr, weight_decay):
        return f"lr={lr:g},weight_decay={weight_decay:g}"

    # The inputs and the labels of the examples at `indices`, one row per index, both built from the records at those indices: 
    # the inputs are [CLS, patch mean] concatenated, the label comes from the metadata of each record.
    def prepare_examples(self, layer_name, target, indices):
        # prepare inputs
        cls_embeddings = self._pooled.stack("cls", layer_name, indices)  # (n, C), n is the number of indices
        patch_embeddings = self._pooled.stack("patch_mean", layer_name, indices)  # (n, C)
        # both lists hold the examples in the same order (LayerwisePooledEmbeddings appends their records in two consecutive lines of one loop iteration), so row i of both is the same example
        inputs = torch.cat([cls_embeddings, patch_embeddings], dim=1)  # (n, 2C)
        assert inputs.shape == (len(indices), 2 * cls_embeddings.shape[1])

        # prepare labels
        labels = []
        for i in indices:
            metadata = self._pooled.records["cls"][layer_name][i]["metadata"]
            labels.append(self.class_index[target][metadata[target]])
        return inputs, torch.tensor(labels, device=inputs.device)

    # Trains one linear head on (train_inputs, train_labels) and returns its metrics on (val_inputs, val_labels) after the final epoch and its per-epoch train and validation loss.
    # The validation loss is a diagnostic only. Both pairs are prepared and standardized in the same way (see run()).
    def fit(self, target, train_inputs, train_labels, val_inputs, val_labels, lr, weight_decay):
        torch.manual_seed(self.seed)  # reseeded for every (fold, grid point), so a training does not depend on the loop order
        generator = torch.Generator().manual_seed(self.seed)
        
        # create probing layer
        num_classes = len(self.class_index[target])
        head = torch.nn.Linear(train_inputs.shape[1], num_classes).to(train_inputs.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

        # CLASS BALANCING
        class_weights = len(train_labels) / (num_classes * torch.bincount(train_labels, minlength=num_classes))  # (K,)
        assert class_weights.shape == (num_classes,)

        train_losses = []
        val_losses = []
        for epoch in range(self._cfg["epochs"]):
            # shuffle train data
            order = torch.randperm(len(train_inputs), generator=generator).to(train_inputs.device)  # the training examples in a new random order
            
            # train epoch
            batch_losses = []
            for start in range(0, len(order), self._cfg["batch_size"]):
                batch_row_idxs = order[start:start + self._cfg["batch_size"]]

                # slice out batch
                batch_rows = train_inputs[batch_row_idxs]  # (batch, D)
                batch_targets = train_labels[batch_row_idxs]  # (batch,)
                assert batch_rows.shape == (len(batch_row_idxs), train_inputs.shape[1]) 
                assert batch_targets.shape == (len(batch_row_idxs),)
                
                # forward and backward passes
                outputs = head(batch_rows)  # (batch, K), one score per class
                assert outputs.shape == (len(batch_row_idxs), num_classes)
                loss = F.cross_entropy(outputs, batch_targets, weight=class_weights)  # scalar (loss computed per example, times its class weight, then the weighted mean over the batch)
                assert loss.shape == ()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.detach())
            train_loss = torch.stack(batch_losses).mean()  # mean of the batch losses
            assert train_loss.shape == ()
            train_losses.append(train_loss.item())
            
            # validation at end of train epoch
            with torch.no_grad():
                # no need for batched loop because validation set is small
                val_outputs = head(val_inputs)  # (n_val, K)
                assert val_outputs.shape == (len(val_inputs), num_classes)
                val_loss = F.cross_entropy(val_outputs, val_labels)
                assert val_loss.shape == ()
            val_losses.append(val_loss.item())

        # final evaluation 
        with torch.no_grad():
            val_outputs = head(val_inputs)
            val_probabilities = torch.softmax(val_outputs, dim=1)
        probabilities = val_probabilities.cpu().numpy()
        y_true = val_labels.cpu().numpy()
        y_pred = probabilities.argmax(axis=1)
        
        # compute metrics
        metrics = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            # macro one-vs-rest AUROC for any number of classes: with one-hot labels, sklearn treats every class as its own binary problem and averages them
            # (the multi_class="ovr" form of roc_auc_score needs a 1-D score when there are two classes, so it would need an if/else)
            "macro_auroc": roc_auc_score(np.eye(num_classes)[y_true], probabilities, average="macro"),
        }

        return metrics, train_losses, val_losses

    def run(self):
        targets_results = {}

        for target in self._cfg["targets"]:
            targets_results[target] = {}

            for layer_name in tqdm(self.layer_names, desc=f"linear probe ({target})"):  # iterate over layers, first to last
                grid = {}

                # initialize grid data structure
                for lr in self._cfg["lrs"]:
                    for weight_decay in self._cfg["weight_decays"]:
                        grid[self._setting_name(lr, weight_decay)] = {"folds": [], "curves": {"train_loss": [], "val_loss": []}}

                # train all folds for all grid positions
                for train_indices, val_indices in self.folds[target]:
                    # prepare train and validation data
                    train_inputs, train_labels = self.prepare_examples(layer_name, target, train_indices)
                    val_inputs, val_labels = self.prepare_examples(layer_name, target, val_indices)

                    # data preprocessing (standardization): 
                    # improves the conditioning number of the embedding cloud (the ratio of its largest to its smallest singular value) i.e. it 
                    # homogenizes the spread of the cloud, which makes the optimization process easier. this matters especially in our context: 
                    # we use the same fixed set of learning rates for every layer, but the embedding clouds after different layers can have vastly different scales and geometryies,
                    # so homogenization is key to one set of of learning rates being meaningful across layers (else we would have to tune them per-layer).
                    mean = train_inputs.mean(dim=0)
                    std = train_inputs.std(dim=0)
                    # DATA LEAKAGE PREVENTION
                    # both folds are standardized with training statistics (no leakage of validation statistics into training set)
                    train_inputs = (train_inputs - mean) / std
                    val_inputs = (val_inputs - mean) / std

                    # train the current fold for each grid position
                    for lr in self._cfg["lrs"]:  # grid axis 1
                        for weight_decay in self._cfg["weight_decays"]:  # grid axis 2
                            metrics, train_losses, val_losses = self.fit(target, train_inputs, train_labels, val_inputs, val_labels, lr, weight_decay)
                            # store results
                            entry = grid[self._setting_name(lr, weight_decay)]
                            entry["folds"].append(metrics)
                            entry["curves"]["train_loss"].append(train_losses)
                            entry["curves"]["val_loss"].append(val_losses)

                # compute mean and std over the folds, and the best grid point by the first metric
                for entry in grid.values():
                    entry["mean"] = {}
                    entry["std"] = {}
                    for metric in METRICS:
                        values = []
                        for fold_metrics in entry["folds"]:
                            values.append(fold_metrics[metric])
                        entry["mean"][metric] = float(np.mean(values))
                        entry["std"][metric] = float(np.std(values))
                best_setting = max(grid, key=lambda setting: grid[setting]["mean"][METRICS[0]])
                targets_results[target][layer_name] = {"grid": grid, "best_setting": best_setting}

        self.results = {"briefing": self.briefing, "targets": targets_results}

    # -------------------------------------------------------------------------------------------------
    # BRIEFING + PLOTTING
    # -------------------------------------------------------------------------------------------------

    def _format_counts(self, counts, class_values):
        total = sum(counts.values())
        parts = []
        for class_value in class_values:
            parts.append(f"{class_value}: {counts[class_value]} ({100 * counts[class_value] / total:.0f}%)")
        return ", ".join(parts)

    # One statistics printout in an ASCII box, once, before the training starts: the plan, the references, the data and the folds, and the feature scale per layer.
    # Returns the same numbers as a dict, which is saved in the results.
    def print_briefing(self, metadatas, group_ids):
        num_grid_points = len(self._cfg["lrs"]) * len(self._cfg["weight_decays"])
        one_input, _ = self.prepare_examples(self.layer_names[0], self._cfg["targets"][0], [0])  # the labels are not needed here, only the shape of an input
        input_dim = one_input.shape[1]

        num_trainings = len(self._cfg["targets"]) * len(self.layer_names) * self._cfg["folds"] * num_grid_points
        total_steps = 0  # optimizer steps of all trainings: every target has its own folds, every fold is trained for every layer and grid point
        for target in self._cfg["targets"]:
            for train_indices, _ in self.folds[target]:
                total_steps += len(self.layer_names) * num_grid_points * self._cfg["epochs"] * math.ceil(len(train_indices) / self._cfg["batch_size"])

        lines = ["LINEAR PROBE: run briefing", "", "PLAN"]
        lines.append(f"targets: {', '.join(self._cfg['targets'])} | layers: {', '.join(self.layer_names)} | input: [CLS, patch mean], {input_dim} values")
        lines.append(f"examples: {self._pooled.num_examples} tiles in {len(set(group_ids))} groups of {self._cfg['leakage_group']} | folds: {self._cfg['folds']} per target (StratifiedGroupKFold on {self._cfg['leakage_group']}, stratified on the target, seed {self.seed})")
        lines.append(f"grid: {len(self._cfg['lrs'])} learning rates x {len(self._cfg['weight_decays'])} weight decays = {num_grid_points} settings | {self._cfg['epochs']} epochs, batch size {self._cfg['batch_size']}")
        lines.append(f"total: {num_trainings} trainings, about {total_steps / 1e6:.1f} M optimizer steps")
        lines.append("")
        lines.append("REFERENCES")
        lines.append(f"seed: {self.seed} | deterministic algorithms: {torch.are_deterministic_algorithms_enabled()} | device: {self._pooled.records['cls'][self.layer_names[0]][0]['embedding'].device}")
        lines.append("")
        lines.append("DATA")
        class_counts = {}
        for target in self._cfg["targets"]:
            counts = Counter()
            for metadata in metadatas:
                counts[metadata[target]] += 1
            class_counts[target] = dict(counts)
            lines.append(f"{target}: {self._format_counts(counts, self.class_index[target])}")
        if len(self._cfg["targets"]) >= 2:  # how the first two targets overlap, to see if they are confounded
            first = self._cfg["targets"][0]
            second = self._cfg["targets"][1]
            second_values = list(self.class_index[second])
            lines.append(f"tiles per {first} (rows) and {second} (columns): {', '.join(map(str, second_values))}")
            for first_value in self.class_index[first]:
                row = []
                for second_value in second_values:
                    count = 0
                    for metadata in metadatas:
                        if metadata[first] == first_value and metadata[second] == second_value:
                            count += 1
                    row.append(str(count))
                lines.append(f"  {first_value}: {', '.join(row)}")
        fold_stats = {}  # per target (every target has its own folds), a list with one entry per fold
        for target in self._cfg["targets"]:
            fold_stats[target] = []
            for fold, (train_indices, val_indices) in enumerate(self.folds[target]):
                train_groups = set()
                for i in train_indices:
                    train_groups.add(group_ids[i])
                val_groups = set()
                for i in val_indices:
                    val_groups.add(group_ids[i])
                train_counts = Counter()
                for i in train_indices:
                    train_counts[metadatas[i][target]] += 1
                val_counts = Counter()
                for i in val_indices:
                    val_counts[metadatas[i][target]] += 1
                fold_stats[target].append({"train_groups": sorted(train_groups), "val_groups": sorted(val_groups), "classes": {"train": dict(train_counts), "val": dict(val_counts)}})
                lines.append(
                    f"fold {fold} | {target} | train {len(train_indices)} tiles, {len(train_groups)} groups: {self._format_counts(train_counts, self.class_index[target])}"
                    f" | val {len(val_indices)} tiles, {len(val_groups)} groups: {self._format_counts(val_counts, self.class_index[target])}"
                )
        lines.append("")
        lines.append("FEATURE SCALE (before standardization)")
        feature_scale = {}
        for layer_name in self.layer_names:
            inputs, _ = self.prepare_examples(layer_name, self._cfg["targets"][0], list(range(self._pooled.num_examples)))  # the labels are not needed here
            norms = inputs.norm(dim=1)
            constant_features = int((inputs.std(dim=0) == 0).sum().item())
            feature_scale[layer_name] = {"norm_mean": norms.mean().item(), "norm_std": norms.std().item(), "constant_features": constant_features}
            lines.append(f"{layer_name}: norm of the input vector {norms.mean().item():.2f} (std {norms.std().item():.2f}), constant features: {constant_features}")

        width = max(len(line) for line in lines)
        print("+" + "-" * (width + 2) + "+")
        for line in lines:
            print("| " + line.ljust(width) + " |")
        print("+" + "-" * (width + 2) + "+")

        return {"class_counts": class_counts, "folds": fold_stats, "feature_scale": feature_scale, "num_trainings": num_trainings, "total_optimizer_steps": total_steps}

    # Headline plot per target: the metrics at the best grid point of every layer. The debug_ plots are two grids per (target, layer): the metric values of every grid point
    # (mean and std over the folds) and the train and validation loss curves of every grid point. Returns {file name: figure}; main() saves them.
    def plot(self):
        figures = {}

        for target, layers_results in self.results["targets"].items():
            figure = Figure(figsize=(8, 5))
            axis = figure.subplots()
            for metric in METRICS:
                means = []
                stds = []
                for layer_name in self.layer_names:
                    best = layers_results[layer_name]["grid"][layers_results[layer_name]["best_setting"]]
                    means.append(best["mean"][metric])
                    stds.append(best["std"][metric])
                axis.errorbar(range(len(self.layer_names)), means, yerr=stds, marker="o", capsize=3, label=metric)
            axis.set_xticks(range(len(self.layer_names)), self.layer_names)
            axis.set_ylim(0, 1.02)
            axis.set_xlabel("layer (increasing depth)")
            axis.set_ylabel("score at the best grid point (mean and std over folds)")
            axis.set_title(f"Linear probe of {target} on [CLS, patch mean]")
            axis.legend(loc="lower right", fontsize=8)
            figures[f"{target}_linear_probe"] = figure

            for layer_name in self.layer_names:
                layer_results = layers_results[layer_name]

                # the value grid: rows are learning rates, columns weight decays, every cell shows the metrics as mean +- std over the folds, the best cell has a red frame
                figure = Figure(figsize=(8, 6))
                axis = figure.subplots()
                primary = []
                for lr in self._cfg["lrs"]:
                    row = []
                    for weight_decay in self._cfg["weight_decays"]:
                        row.append(layer_results["grid"][self._setting_name(lr, weight_decay)]["mean"][METRICS[0]])
                    primary.append(row)
                axis.imshow(primary, vmin=0, vmax=1)
                for row_index, lr in enumerate(self._cfg["lrs"]):
                    for column_index, weight_decay in enumerate(self._cfg["weight_decays"]):
                        entry = layer_results["grid"][self._setting_name(lr, weight_decay)]
                        cell_lines = []
                        for metric in METRICS:
                            cell_lines.append(f"{metric}\n{entry['mean'][metric]:.3f} +- {entry['std'][metric]:.3f}")
                        axis.text(column_index, row_index, "\n".join(cell_lines), ha="center", va="center", fontsize=6, bbox={"facecolor": "white", "alpha": 0.75, "pad": 1})
                        if self._setting_name(lr, weight_decay) == layer_results["best_setting"]:
                            axis.add_patch(Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3))
                axis.set_xticks(range(len(self._cfg["weight_decays"])), [f"{value:g}" for value in self._cfg["weight_decays"]])
                axis.set_yticks(range(len(self._cfg["lrs"])), [f"{value:g}" for value in self._cfg["lrs"]])
                axis.set_xlabel("weight decay")
                axis.set_ylabel("learning rate")
                axis.set_title(f"{target}, {layer_name}: metrics per grid point (mean +- std over folds)")
                figures[f"debug_{target}_{layer_name}_values"] = figure

                # the curve grid: one small plot per grid point with the train and validation loss per epoch (mean over the folds, band from the min to the max fold)
                figure = Figure(figsize=(9, 7))
                axes = figure.subplots(len(self._cfg["lrs"]), len(self._cfg["weight_decays"]), sharex=True, sharey=True, squeeze=False)
                for row_index, lr in enumerate(self._cfg["lrs"]):
                    for column_index, weight_decay in enumerate(self._cfg["weight_decays"]):
                        curves = layer_results["grid"][self._setting_name(lr, weight_decay)]["curves"]
                        axis = axes[row_index][column_index]
                        for name, label in (("train_loss", "train"), ("val_loss", "validation")):
                            per_fold = np.array(curves[name])  # (folds, epochs)
                            epochs = range(1, per_fold.shape[1] + 1)
                            axis.plot(epochs, per_fold.mean(axis=0), label=label)
                            axis.fill_between(epochs, per_fold.min(axis=0), per_fold.max(axis=0), alpha=0.2)
                        axis.set_title(self._setting_name(lr, weight_decay), fontsize=7)
                        axis.tick_params(labelsize=6)
                axes[0][0].legend(fontsize=6)
                figure.supxlabel("epoch")
                figure.supylabel("cross-entropy loss (train: class-weighted, validation: unweighted)")
                figure.suptitle(f"{target}, {layer_name}: loss curves per grid point")
                figures[f"debug_{target}_{layer_name}_curves"] = figure

        return figures
