import torch

class OnlineClusterWeighting:
    def __init__(self, weights, cluster_ids, num_clusters, ema_decay):
        self.weights = weights.clone()
        self.cluster_ids = cluster_ids
        self.ema_decay = ema_decay
        # accumulation state (reset after each update)
        self.loss_sum = torch.zeros(num_clusters, dtype=torch.float64)  # per cluster sum of accumulated losses
        self.loss_count = torch.zeros(num_clusters, dtype=torch.long)  # per cluster number of accumulated losses

    def accumulate(self, cluster_ids, losses):
        losses = losses.detach().to(device=self.loss_sum.device, dtype=self.loss_sum.dtype)
        self.loss_sum.scatter_add_(0, cluster_ids, losses)  # add new losses into the right cluster-buckets (= a scatter driven by ids) 
        self.loss_count.scatter_add_(0, cluster_ids, torch.ones_like(cluster_ids))  # same operation for counts (scatter is the elegant function here as well)

    def update_weights(self):
        average_cluster_losses = torch.zeros_like(self.loss_sum)  # create slots for every cluster
        
        observed = self.loss_count > 0  # only update clusters that received updates during accumulation - others would divide by zero
        average_cluster_losses[observed] = self.loss_sum[observed] / self.loss_count[observed]  # per cluster the average loss seen in the current accumulation window
        
        per_tile = average_cluster_losses[self.cluster_ids]  # we have computed losses per cluster, but weights are per tile i.e. expand information into per-tile shape
        distribution = per_tile  / per_tile.sum()  # normalize to probability distribution s.t. interpolation is between comparable scales
        
        self.weights.lerp_(distribution, 1 - self.ema_decay)  # EMA style interpolation
        
        # reset accumulation state 
        self.loss_sum.zero_()
        self.loss_count.zero_()