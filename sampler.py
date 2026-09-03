import torch
import numpy as np
from torchvision.utils import make_grid
import wandb


HISTOGRAM_BUCKETS = 50
IMAGE_GRID_TOP_K = 4
IMAGE_GRID_EVERY_UPDATES = 1

class OnlineClusterWeighting:
    def __init__(self, weights, cluster_ids, num_clusters, ema_decay):
        self.weights = weights.clone()
        self.cluster_ids = cluster_ids
        self.ema_decay = ema_decay
        # accumulation state (reset after each update)
        self.loss_sum = torch.zeros(num_clusters, dtype=torch.float64)  # per cluster sum of accumulated losses
        self.loss_count = torch.zeros(num_clusters, dtype=torch.long)  # per cluster number of accumulated losses
        self.image_by_cluster = {}
        self.update_count = 0

    def accumulate(self, cluster_ids, losses, images):
        losses = losses.detach().to(device=self.loss_sum.device, dtype=self.loss_sum.dtype)
        self.loss_sum.scatter_add_(0, cluster_ids, losses)  # add new losses into the right cluster-buckets (= a scatter driven by ids) 
        self.loss_count.scatter_add_(0, cluster_ids, torch.ones_like(cluster_ids))  # same operation for counts (scatter is the elegant function here as well)
        if (self.update_count + 1) % IMAGE_GRID_EVERY_UPDATES == 0:
            for cluster_id, image in zip(cluster_ids.tolist(), images):
                if cluster_id in self.image_by_cluster:
                    continue
                else:
                    clamped_image = image.clamp(0, 1)
                    scaled_image = clamped_image * 255
                    image_uint8 = scaled_image.to(torch.uint8)
                    self.image_by_cluster[cluster_id] = image_uint8

    def _log_cluster_loss_distribution(self, wandb_run, step, average_cluster_losses):
        normalized_losses = average_cluster_losses / average_cluster_losses.sum()
        wandb_run.log({"sampler/cluster_loss_distribution": wandb.Histogram(
            np_histogram=np.histogram(normalized_losses.numpy(), bins=HISTOGRAM_BUCKETS)
        )}, step=step)

    def _log_cluster_observation_count(self, wandb_run, step):
        wandb_run.log({"sampler/cluster_observation_count": wandb.Histogram(
            np_histogram=np.histogram(self.loss_count.numpy(), bins=HISTOGRAM_BUCKETS)
        )}, step=step)

    def _log_effective_sample_size(self, wandb_run, step):
        effective_sample_size = self.weights.sum().square() / self.weights.square().sum()
        wandb_run.log({"sampler/effective_sample_size": effective_sample_size.item()}, step=step)

    def _log_cluster_image_grid(self, wandb_run, step, average_cluster_losses, observed):
        # need to filter on observed clusters in order to not try to load images for clusters that did not see any sampling this window
        observed_clusters = observed.nonzero(as_tuple=True)[0]
        loss_order = torch.argsort(average_cluster_losses[observed_clusters])
        easy_clusters = observed_clusters[loss_order[:IMAGE_GRID_TOP_K]]
        hard_clusters = observed_clusters[loss_order[-IMAGE_GRID_TOP_K:]].flip(0)
        selected_clusters = torch.cat((hard_clusters, easy_clusters))
        images = [self.image_by_cluster[int(cluster_id)] for cluster_id in selected_clusters]
        grid = make_grid(torch.stack(images), nrow=IMAGE_GRID_TOP_K, padding=2)
        labels = [f"hard {int(cluster_id)}: {average_cluster_losses[cluster_id]:.3f}" for cluster_id in hard_clusters]
        labels += [f"easy {int(cluster_id)}: {average_cluster_losses[cluster_id]:.3f}" for cluster_id in easy_clusters]
        wandb_run.log({"sampler/cluster_image_grid": wandb.Image(grid, caption=" | ".join(labels))}, step=step)

    def update_weights(self, step, wandb_run):
        self.update_count += 1

        average_cluster_losses = torch.zeros_like(self.loss_sum)  # create slots for every cluster
        
        observed = self.loss_count > 0  # only update clusters that received updates during accumulation - others would divide by zero
        average_cluster_losses[observed] = self.loss_sum[observed] / self.loss_count[observed]  # per cluster the average loss seen in the current accumulation window
        
        per_tile = average_cluster_losses[self.cluster_ids]  # we have computed losses per cluster, but weights are per tile i.e. expand information into per-tile shape
        distribution = per_tile  / per_tile.sum()  # normalize to probability distribution s.t. interpolation is between comparable scales
        
        self.weights.lerp_(distribution, 1 - self.ema_decay)  # EMA style interpolation

        self._log_cluster_loss_distribution(wandb_run, step, average_cluster_losses)
        self._log_cluster_observation_count(wandb_run, step)
        self._log_effective_sample_size(wandb_run, step)
        if self.update_count % IMAGE_GRID_EVERY_UPDATES == 0:
            self._log_cluster_image_grid(wandb_run, step, average_cluster_losses, observed)
        
        # reset accumulation state 
        self.loss_sum.zero_()
        self.loss_count.zero_()
        self.image_by_cluster.clear()