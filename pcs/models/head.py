import torch
import torch.nn as nn
import torch.nn.functional as F
from pcs.utils.torchutils import grad_reverse
from torch.autograd import Function
from torchvision import models


class CosineClassifier(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        self.normalize_fc()

        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x)
        x_out = x_out / self.temp

        return x_out

    def normalize_fc(self):
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, eps=1e-12, dim=1)

    @torch.no_grad()
    def compute_discrepancy(self):
        self.normalize_fc()
        W = self.fc.weight.data
        D = torch.mm(W, W.transpose(0, 1))
        D_mask = 1 - torch.eye(self.num_class).cuda()
        return torch.sum(D * D_mask).item()
