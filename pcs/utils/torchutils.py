import logging
import os
import random
import shutil
import sys
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function

# Setup


def set_seed(seed=1234, determine=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if determine:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Tensor


def expand_1d(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def isin(ar1, ar2):
    # for every element of ar2, is ar2 in ar1
    # return shape same to ar1
    return (ar1[..., None] == ar2).any(-1)


def dot(x, y):
    return torch.sum(x * y, dim=-1)


def contrastive_sim(instances, proto=None, tao=0.05):
    # prob_matrix [bs, dim]
    # proto_dim [nums, dim]
    if proto is None:
        proto = instances
    ins_ext = instances.unsqueeze(1).repeat(1, proto.size(0), 1)
    sim_matrix = torch.exp(dot(ins_ext, proto) / tao)
    return sim_matrix


def contrastive_sim_z(instances, proto=None, tao=0.05):
    sim_matrix = contrastive_sim(instances, proto, tao)
    return torch.sum(sim_matrix, dim=-1)


def contrastive_prob(instances, proto=None, tao=0.05):
    sim_matrix = contrastive_sim(instances, proto, tao)
    return sim_matrix / torch.sum(sim_matrix, dim=-1).unsqueeze(-1)


def pairwise_distance_2(input_1, input_2):
    assert input_1.size(1) == input_2.size(1)
    dis_vec = input_1.unsqueeze(1) - input_2
    dis = torch.norm(dis_vec, dim=2)
    return dis


# nn


def weights_init(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu"
            )
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)


def split_params_by_name(model, name):
    if not isinstance(name, list):
        name = [name]
    with_name = []
    without_name = []
    for key, param in model.named_parameters():
        if not param.requires_grad:
            continue

        in_key = False
        for n in name:
            in_key = in_key | (n in key)

        if in_key:
            with_name.append(param)
        else:
            without_name.append(param)
    return with_name, without_name


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# nn.functional


def entropy(x, eps=1e-5):
    p = F.softmax(x, dim=-1)
    entropy = -torch.mean(torch.sum(p * torch.log(p + eps), 1))
    return entropy


def pseudo_mask(x, thres=0.95):
    prob = F.softmax(x, dim=1)
    max_prob, pred = prob.max(dim=1)
    mask = max_prob > thres
    return mask


def pseudo_label_loss(x, thres=0.95, aux=True, y=None, mask=None, num_class=10):
    if mask is None:
        mask = [True] * len(x)
        mask = torch.tensor(mask)
    # update mask
    prob = F.softmax(x, dim=1)
    max_prob, pred = prob.max(dim=1)
    mask[max_prob < thres] = False
    num_thres = mask.sum().item()

    # calculate loss
    out_thres, pred_thres = x[mask], pred[mask]
    if num_thres == 0:
        loss = torch.tensor(0)
    else:
        loss = F.cross_entropy(out_thres, pred_thres)

    if aux:
        num_select_per_class = [0] * num_class
        num_correct_per_class = [0] * num_class
        for i in range(num_class):
            num_select_per_class[i] += (pred_thres == i).sum().item()
            if y is not None:
                num_correct_per_class[i] += (
                    ((pred_thres == i) & (pred_thres.eq(y[mask]))).sum().item()
                )

        if y is not None:
            num_correct = pred_thres.eq(y[mask]).sum().item()
        else:
            num_correct = -1

        ret_aux = {
            "num_select": num_thres,
            "num_correct": num_correct,
            "num_select_per_class": num_select_per_class,
            "num_correct_per_class": num_correct_per_class,
            "mask": mask,
        }
        return loss, ret_aux
    else:
        return loss


# optim


def lr_scheduler_invLR(optimizer, gamma=0.0001, power=0.75):
    def lmbda(iter):
        return (1 + gamma * iter) ** (-power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)


def get_lr(optimizer, g_id=0):
    return optimizer.param_groups[g_id]["lr"]


# utils


def copy_checkpoint(
    folder="./", filename="checkpoint.pth.tar", copyname="copy.pth.tar"
):
    shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, copyname))


def save_checkpoint(state, is_best=False, folder="./", filename="checkpoint.pth.tar"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        copy_checkpoint(folder, filename, "model_best.pth.tar")


def load_state_dict(model, model_dict):
    model_dict = model.state_dict()
    updated_dict = {k: v for k, v in model_dict.items() if k in model_dict.keys()}
    model_dict.update(updated_dict)
    model.load_state_dict(model_dict)
    return len(updated_dict.keys())


def print_cuda_statistics(nvidia_smi=True, output=print):
    output(f"Python VERSION: {sys.version}")
    output(f"pytorch VERSION: {torch.__version__}")
    output(f"CUDA VERSION: {torch.version.cuda}")
    output(f"CUDNN VERSION: {torch.backends.cudnn.version()}")
    output(f"Device NAME: {torch.cuda.get_device_name(0)}")
    output(f"Number CUDA Devices: {torch.cuda.device_count()}")
    output(f"Available devices: {torch.cuda.device_count()}")
    output(f"current CUDA Device: {torch.cuda.current_device()}")

    if nvidia_smi:
        print("nvidia-smi:")
        call(
            [
                "nvidia-smi",
                "--format=csv",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
            ]
        )


def log_tensor(t, name="", print_tensor=False):
    print(
        f"Tensor {name}:\n\ttype: {t.type()}\n\tsize {t.shape}\n\tdim: {t.dim()}\n\tdevice: {t.device}\n\tnelement: {t.nelement()}\n\telem_size: {t.element_size()}\n\tsize in mem: {t.nelement() * t.element_size()} Bytes\n\tgrad_fn: {t.grad_fn}\n\tgrad: {t.grad}"
    )
    if print_tensor:
        print(t)


def model_params_num(model):
    return sum(torch.numel(parameter) for parameter in model.parameters())


def one_hot(label):
    N = label.size(0)
    num_classes = label.unique().size(0)
    one_hot = torch.zeros(N, num_classes).long()
    one_hot.scatter_(
        dim=1,
        index=torch.unsqueeze(label, dim=1),
        src=torch.ones(N, num_classes).long(),
    )


def top_k_for_each_class(pred, prob, num_class):
    ind = torch.arange(len(pred)).long()
    pred_ret = torch.zeros_like(pred).long().cuda() - 1
    for i in range(num_class):
        class_mask = pred == i
        num_c = class_mask.sum()
        num_c = class_mask.sum()
        if num_c == 0:
            continue
        prob_class = prob[class_mask]
        ind_class = ind[class_mask]
        prob_topk, ind_topk = prob_class.topk(min(5, num_c))
        ind_topk = ind_class[ind_topk]
        pred_ret[ind_topk] = i
    return pred_ret


# MIM


class MomentumSoftmax:
    def __init__(self, num_class, m=1):
        self.softmax_vector = torch.zeros(num_class).detach() + 1.0 / num_class
        self.m = m
        self.num = m

    def update(self, mean_softmax, num=1):
        self.softmax_vector = (
            (self.softmax_vector * self.num) + mean_softmax * num
        ) / (self.num + num)
        self.num += num

    def reset(self):
        # print(self.softmax_vector)
        self.num = self.m
