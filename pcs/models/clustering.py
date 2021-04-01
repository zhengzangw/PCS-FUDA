import logging
import time
from collections import Counter

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from faiss import Kmeans as faiss_Kmeans
from tqdm import tqdm

DEFAULT_KMEANS_SEED = 1234


class Kmeans(object):
    def __init__(
        self, k_list, data, epoch=0, init_centroids=None, frozen_centroids=False
    ):
        """
        Performs many k-means clustering.

        Args:
            data (np.array N * dim): data to cluster
        """
        super().__init__()
        self.k_list = k_list
        self.data = data
        self.d = data.shape[-1]
        self.init_centroids = init_centroids
        self.frozen_centroids = frozen_centroids

        self.logger = logging.getLogger("Kmeans")
        self.debug = False
        self.epoch = epoch + 1

    def compute_clusters(self):
        """compute cluster

        Returns:
            torch.tensor, list: clus_labels, centroids
        """
        data = self.data
        labels = []
        centroids = []

        tqdm_batch = tqdm(total=len(self.k_list), desc="[K-means]")
        for k_idx, each_k in enumerate(self.k_list):
            seed = k_idx * self.epoch + DEFAULT_KMEANS_SEED
            kmeans = faiss_Kmeans(
                self.d,
                each_k,
                niter=40,
                verbose=False,
                spherical=True,
                min_points_per_centroid=1,
                max_points_per_centroid=10000,
                gpu=True,
                seed=seed,
                frozen_centroids=self.frozen_centroids,
            )

            kmeans.train(data, init_centroids=self.init_centroids)

            _, I = kmeans.index.search(data, 1)
            labels.append(I.squeeze(1))
            C = kmeans.centroids
            centroids.append(C)

            tqdm_batch.update()
        tqdm_batch.close()

        labels = np.stack(labels, axis=0)

        return labels, centroids


def torch_kmeans(k_list, data, init_centroids=None, seed=0, frozen=False):
    if init_centroids is not None:
        init_centroids = init_centroids.cpu().numpy()
    km = Kmeans(
        k_list,
        data.cpu().detach().numpy(),
        epoch=seed,
        frozen_centroids=frozen,
        init_centroids=init_centroids,
    )
    clus_labels, centroids_npy = km.compute_clusters()
    clus_labels = torch.from_numpy(clus_labels).long().cuda()
    centroids = []
    for c in centroids_npy:
        centroids.append(torch.from_numpy(c).cuda())
    # compute phi
    clus_phi = []
    for i in range(len(k_list)):
        clus_phi.append(compute_variance(data, clus_labels[i], centroids[i]))

    return clus_labels, centroids, clus_phi


# variance


@torch.no_grad()
def compute_variance(
    data, cluster_labels, centroids, alpha=10, debug=False, num_class=None
):
    """compute variance for proto

    Args:
        data (torch.Tensor): data with shape [n, dim] 
        cluster_labels (torch.Tensor): cluster labels of [n]
        centroids (torch.Tensor): cluster centroids [k, ndim]
        alpha (int, optional): Defaults to 10.
        debug (bool, optional): Defaults to False.

    Returns:
        [type]: [description]
    """

    k = len(centroids) if num_class is None else num_class
    phis = torch.zeros(k)
    for c in range(k):
        cluster_points = data[cluster_labels == c]
        c_len = len(cluster_points)
        if c_len == 0:
            phis[c] = -1
        elif c_len == 1:
            phis[c] = 0.05
        else:
            phis[c] = torch.sum(torch.norm(cluster_points - centroids[c], dim=1)) / (
                c_len * np.log(c_len + alpha)
            )
            if phis[c] < 0.05:
                phis[c] = 0.05

    if debug:
        print("size-phi:", end=" ")
        for i in range(k):
            size = (cluster_labels == i).sum().item()
            print(f"{size}[phi={phis[i].item():.3f}]", end=", ")
        print("\n")

    return phis
