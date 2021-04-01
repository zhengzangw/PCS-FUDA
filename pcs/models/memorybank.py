import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MemoryBank(object):
    """For efficiently computing the background vectors."""

    def __init__(self, size, dim):
        """generate random memory bank

        Args:
            size (int): length of the memory bank
            dim (int): dimension of the memory bank features
            device_ids (list): gpu lists
        """
        self._bank = self._create(size, dim)

    def _create(self, size, dim):
        """generate randomized features
        """
        # initialize random weights
        mb_init = torch.rand(size, dim, device=torch.device("cuda"))
        std_dev = 1.0 / np.sqrt(dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        # L2 normalize so that the norm is 1
        mb_init = F.normalize(mb_init)
        return mb_init.detach()  # detach so its not trainable

    def as_tensor(self):
        return self._bank

    def at_idxs(self, idxs):
        return torch.index_select(self._bank, 0, idxs)

    def update(self, indices, data_memory):
        data_dim = data_memory.size(1)
        data_memory = data_memory.detach()
        indices = indices.unsqueeze(1).repeat(1, data_dim)

        self._bank = self._bank.scatter_(0, indices, data_memory)
