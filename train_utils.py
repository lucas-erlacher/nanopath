# Training-only setup and loader lifecycle helpers.
# This module coordinates offline metadata with the optional online sampler.

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from online_sampler import OnlineSampler


def create_train_loader(dataset, weights, sample_count, loader_kwargs):
    sampler = WeightedRandomSampler(weights, sample_count, replacement=True)
    return DataLoader(dataset, sampler=sampler, **loader_kwargs)


def shutdown_loader_workers(loader):
    if loader._iterator is not None:
        loader._iterator._shutdown_workers()
        loader._iterator = None


############    SAMPLER    ############


def setup_sampler(cfg, dataset, batch_size, loader_kwargs):
    online_cfg = cfg["online_cluster_weighting"]
    online_sampler = None
    if bool(online_cfg["enabled"]):
        online_sampler = OnlineSampler(
            dataset.offline_sampler.weights,
            torch.as_tensor(dataset.sampling_metadata.cluster_ids, dtype=torch.long),
            int(cfg["prune"]["num_clusters"]),
            float(online_cfg["ema_decay"]),
            batch_size * int(online_cfg["update_every_steps"]),
        )
    window_samples = online_sampler.window_samples if online_sampler is not None else len(dataset)
    loader = create_train_loader(dataset, dataset.offline_sampler.weights, window_samples, loader_kwargs)
    return online_sampler, window_samples, loader


def update_sampler_after_window(sampler, step, wandb_run, loader, dataset, loader_kwargs):
    sampler.update(step, wandb_run)
    shutdown_loader_workers(loader)
    return create_train_loader(dataset, sampler.weights, sampler.window_samples, loader_kwargs)
