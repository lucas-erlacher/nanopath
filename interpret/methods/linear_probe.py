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
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from interpret.utils import standardize

METRICS = ("balanced_accuracy", "macro_f1", "macro_auroc")  # the first one selects the best grid point


class LinearProbe:
    def __init__(self, datasets, device, seed, method_cfg):
        self._datasets = datasets  # {layer name: ActivationDataset}, see interpret.py
        self.device = device  # where the linear head lives (the datasets already hand out tensors on it)
        self.seed = seed
        self._cfg = method_cfg
        self.results = None

        self.layer_names = list(datasets)

        # the labels (integers) and their names are the same in the dataset of every layer, so the first one gives them all
        first_dataset = list(datasets.values())[0]
        self.labels = first_dataset.get_block("labels", to_device=False)  # all labels, they stay on the CPU: the folds (sklearn) and the briefing work on the host
        self.label_names = first_dataset.label_names  
        group_ids = self.labels[self._cfg["leakage_group"]].numpy()

        # pre-compute folds (once per target)
        # DATA LEAKAGE PREVENTION
        # CLASS BALANCING
        self.folds = {}
        for target in self._cfg["targets"]:
            splitter = StratifiedGroupKFold(n_splits=self._cfg["folds"], shuffle=True, random_state=seed)
            self.folds[target] = []
            for train_indices, val_indices in splitter.split(list(range(len(first_dataset))), self.labels[target].numpy(), groups=group_ids):
                self.folds[target].append((train_indices.tolist(), val_indices.tolist()))

        self.briefing = self.print_briefing(first_dataset)

    # -------------------------------------------------------------------------------------------------
    # TRAINING + EVALUATION
    # -------------------------------------------------------------------------------------------------

    def _setting_name(self, lr, weight_decay):
        return f"lr={lr:g},weight_decay={weight_decay:g}"

    # The inputs and the labels of one batch of an ActivationDataset: the inputs are [CLS, patch mean] concatenated, the labels are the integers of the target.
    def prepare_batch(self, batch, target):
        inputs = torch.cat([batch["cls"], batch["patch_mean"]], dim=1)  # (batch, 2C)
        labels = batch["labels"][target]  # (batch,)
        assert inputs.shape == (len(labels), 2 * batch["cls"].shape[1])
        return inputs, labels

    # Trains one linear head on the training subset and returns its metrics on the validation subset after the final epoch and its per-epoch train and validation loss.
    # The validation loss is a diagnostic only. Both subsets are prepared and standardized in the same way (with the training statistics mean and std, see run()).
    def fit(self, target, dataset, train_indices, val_indices, mean, std, class_weights, lr, weight_decay):
        torch.manual_seed(self.seed)  # reseeded for every (fold, grid point), so a training does not depend on the loop order
        generator = torch.Generator().manual_seed(self.seed)
        train_loader = dataset.loader(train_indices, self._cfg["batch_size"], shuffle=True, generator=generator)  # the training examples in a new random order every epoch, batches already on the device
        val_loader = dataset.loader(val_indices, self._cfg["batch_size"])

        # create probing layer
        num_classes = len(class_weights)
        head = torch.nn.Linear(mean.shape[0], num_classes).to(self.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)


        train_losses = []
        val_losses = []
        for epoch in range(self._cfg["epochs"]):
            # train epoch
            batch_losses = []
            for batch in train_loader:
                inputs, targets = self.prepare_batch(batch, target)  # (batch, D), (batch,)
                inputs = standardize(inputs, mean, std)
                assert targets.shape == (len(inputs),)

                # forward and backward passes
                outputs = head(inputs)  # (batch, K), one score per class
                assert outputs.shape == (len(inputs), num_classes)
                loss = F.cross_entropy(outputs, targets, weight=class_weights)  # scalar (loss computed per example, times its class weight, then the weighted mean over the batch)
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
                batch_losses = []
                val_outputs = []  # collected for the metrics of the last epoch
                val_targets = [] # collected for the metrics of the last epoch
                for batch in val_loader:
                    inputs, targets = self.prepare_batch(batch, target)  # (batch, D), (batch,)
                    inputs = standardize(inputs, mean, std)
                    assert targets.shape == (len(inputs),)

                    # forward pass
                    outputs = head(inputs)  # (batch, K), one score per class
                    assert outputs.shape == (len(inputs), num_classes)
                    loss = F.cross_entropy(outputs, targets)  # scalar, unweighted: the evaluation is not balanced
                    assert loss.shape == ()
                    batch_losses.append(loss)
                    val_outputs.append(outputs)
                    val_targets.append(targets)
                val_loss = torch.stack(batch_losses).mean()  # mean of the batch losses
                assert val_loss.shape == ()
                val_outputs = torch.cat(val_outputs)  # (n_val, K)
                val_targets = torch.cat(val_targets)  # (n_val,)
                assert val_outputs.shape == (len(val_indices), num_classes)
            val_losses.append(val_loss.item())

        # final evaluation: the head has not changed since the validation of the last epoch, so those outputs are the final outputs
        probabilities = torch.softmax(val_outputs, dim=1).cpu().numpy()
        y_true = val_targets.cpu().numpy()
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
                dataset = self._datasets[layer_name]
                grid = {}

                # initialize grid data structure
                for lr in self._cfg["lrs"]:
                    for weight_decay in self._cfg["weight_decays"]:
                        grid[self._setting_name(lr, weight_decay)] = {"folds": [], "curves": {"train_loss": [], "val_loss": []}}

                # train all folds for all grid positions
                for fold, (train_indices, val_indices) in enumerate(self.folds[target]):
                    # data preprocessing (standardization): 
                    # improves the conditioning number of the embedding cloud (the ratio of its largest to its smallest singular value) i.e. it 
                    # homogenizes the spread of the cloud, which makes the optimization process easier. this matters especially in our context: 
                    # we use the same fixed set of learning rates for every layer, but the embedding clouds after different layers can have vastly different scales and geometryies,
                    # so homogenization is key to one set of of learning rates being meaningful across layers (else we would have to tune them per-layer).
                    # load the whole training fold at once (the rows of the whole blocks), for the statistics and the class weights
                    train_inputs = torch.cat([dataset.get_block("activations", "cls")[train_indices], dataset.get_block("activations", "patch_mean")[train_indices]], dim=1)
                    train_labels = dataset.get_block("labels", target)[train_indices]
                    mean = train_inputs.mean(dim=0)
                    std = train_inputs.std(dim=0)
                    # DATA LEAKAGE PREVENTION
                    # fit() standardizes every batch of both folds with these training statistics (no leakage of validation statistics into training set)

                    # CLASS BALANCING
                    # fit() weights the loss of every class with these weights, which come from the training fold only
                    num_classes = len(self.label_names[target])
                    class_weights = len(train_labels) / (num_classes * torch.bincount(train_labels, minlength=num_classes))  # (K,)
                    assert class_weights.shape == (num_classes,)

                    # train the current fold for each grid position
                    for lr in self._cfg["lrs"]:  # grid axis 1
                        for weight_decay in self._cfg["weight_decays"]:  # grid axis 2
                            start_time = time.perf_counter()
                            metrics, train_losses, val_losses = self.fit(target, dataset, train_indices, val_indices, mean, std, class_weights, lr, weight_decay)
                            tqdm.write(f"{target} | {layer_name} | fold {fold} | {self._setting_name(lr, weight_decay)}: trained in {time.perf_counter() - start_time:.1f} s (balanced accuracy {metrics['balanced_accuracy']:.3f})")
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
    def print_briefing(self, first_dataset):
        num_grid_points = len(self._cfg["lrs"]) * len(self._cfg["weight_decays"])
        input_dim = 2 * first_dataset.get_block("activations", "cls").shape[1]  # [CLS, patch mean]
        group_field = self._cfg["leakage_group"]
        group_names = self.label_names[group_field]

        num_trainings = len(self._cfg["targets"]) * len(self.layer_names) * self._cfg["folds"] * num_grid_points
        total_steps = 0  # optimizer steps of all trainings: every target has its own folds, every fold is trained for every layer and grid point
        for target in self._cfg["targets"]:
            for train_indices, _ in self.folds[target]:
                total_steps += len(self.layer_names) * num_grid_points * self._cfg["epochs"] * math.ceil(len(train_indices) / self._cfg["batch_size"])

        lines = ["LINEAR PROBE: run briefing", "", "PLAN"]
        lines.append(f"targets: {', '.join(self._cfg['targets'])} | layers: {', '.join(self.layer_names)} | input: [CLS, patch mean], {input_dim} values")
        lines.append(f"examples: {len(first_dataset)} tiles in {len(group_names)} groups of {self._cfg['leakage_group']} | folds: {self._cfg['folds']} per target (StratifiedGroupKFold on {self._cfg['leakage_group']}, stratified on the target, seed {self.seed})")
        lines.append(f"grid: {len(self._cfg['lrs'])} learning rates x {len(self._cfg['weight_decays'])} weight decays = {num_grid_points} settings | {self._cfg['epochs']} epochs, batch size {self._cfg['batch_size']}")
        lines.append(f"total: {num_trainings} trainings, about {total_steps / 1e6:.1f} M optimizer steps")
        lines.append("")
        lines.append("REFERENCES")
        lines.append(f"seed: {self.seed} | deterministic algorithms: {torch.are_deterministic_algorithms_enabled()} | device: {self.device}")
        lines.append("")
        lines.append("DATA")
        class_counts = {}
        for target in self._cfg["targets"]:
            counts = Counter(self.label_names[target][k] for k in self.labels[target].tolist())
            class_counts[target] = dict(counts)
            lines.append(f"{target}: {self._format_counts(counts, self.label_names[target])}")
        if len(self._cfg["targets"]) >= 2:  # how the first two targets overlap, to see if they are confounded
            first = self._cfg["targets"][0]
            second = self._cfg["targets"][1]
            lines.append(f"tiles per {first} (rows) and {second} (columns): {', '.join(map(str, self.label_names[second]))}")
            for first_index, first_name in enumerate(self.label_names[first]):
                row = []
                for second_index in range(len(self.label_names[second])):
                    count = ((self.labels[first] == first_index) & (self.labels[second] == second_index)).sum().item()
                    row.append(str(count))
                lines.append(f"  {first_name}: {', '.join(row)}")
        fold_stats = {}  # per target (every target has its own folds), a list with one entry per fold
        for target in self._cfg["targets"]:
            fold_stats[target] = []
            for fold, (train_indices, val_indices) in enumerate(self.folds[target]):
                train_groups = sorted({group_names[k] for k in self.labels[group_field][train_indices].tolist()})
                val_groups = sorted({group_names[k] for k in self.labels[group_field][val_indices].tolist()})
                train_counts = Counter(self.label_names[target][k] for k in self.labels[target][train_indices].tolist())
                val_counts = Counter(self.label_names[target][k] for k in self.labels[target][val_indices].tolist())
                fold_stats[target].append({"train_groups": train_groups, "val_groups": val_groups, "classes": {"train": dict(train_counts), "val": dict(val_counts)}})
                lines.append(
                    f"fold {fold} | {target} | train {len(train_indices)} tiles, {len(train_groups)} groups: {self._format_counts(train_counts, self.label_names[target])}"
                    f" | val {len(val_indices)} tiles, {len(val_groups)} groups: {self._format_counts(val_counts, self.label_names[target])}"
                )
        lines.append("")
        lines.append("FEATURE SCALE (before standardization)")
        feature_scale = {}
        for layer_name, dataset in self._datasets.items():
            inputs = torch.cat([dataset.get_block("activations", "cls"), dataset.get_block("activations", "patch_mean")], dim=1)  # the whole layer at once, [CLS, patch mean] like in prepare_batch
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
