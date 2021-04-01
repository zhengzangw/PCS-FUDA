import collections
import os
import random
import shutil
import socket

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

# image_list


def create_image_label(image_list):
    image_index = [x.split(" ")[0] for x in open(image_list)]
    label_list = np.array([int(x.split(" ")[1].strip()) for x in open(image_list)])
    return image_index, label_list


def get_class_map(image_list):
    class_map = {}
    for x in open(image_list):
        key = int(x.split(" ")[1].strip())
        if key not in class_map:
            class_map[key] = x.split(" ")[0].split("/")[-2]
    class_map = collections.OrderedDict(sorted(class_map.items()))
    return class_map


def get_class_num(image_list):
    # return len(get_class_map(image_list))
    return max(list(get_class_map(image_list).keys())) + 1


def describe_image_list(image_list, save_graph=False, label_name=True, is_sort=False):
    _, label_list = create_image_label(image_list)
    label_cnt = np.bincount(label_list)
    print(
        f"""Image list \"{image_list}\":
    \tTotal instances: {len(label_list)}
    \tTotal class: {len(label_cnt)}
    \tmax # of class: {np.max(label_cnt)}
    \tmin # of class: {np.min(label_cnt)}
    \tmean # of class: {np.mean(label_cnt)}
    \tmedian # of class: {np.median(label_cnt)}
    \tvar: {np.var(label_cnt)}"""
    )


def get_fewshot_index(lbd_dataset, whl_dataset):
    lbd_imgs = lbd_dataset.imgs
    whl_imgs = whl_dataset.imgs
    fewshot_indices = [whl_imgs.index(path) for path in lbd_imgs]
    fewshot_labels = lbd_dataset.labels
    return fewshot_indices, fewshot_labels

class Imagelists(torch.utils.data.Dataset):
    def __init__(
        self,
        image_list,
        root,
        transform=None,
        target_transform=None,
        keep_in_mem=False,
        ret_index=False,
    ):
        # print(image_list)
        imgs, labels = create_image_label(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.ret_index = ret_index
        self.keep_in_mem = keep_in_mem
        self.loader = pil_loader

        # keep in mem
        if self.keep_in_mem:
            images = []
            for index in range(len(self.imgs)):
                path = os.path.join(self.root, self.imgs[index])
                img = self.loader(path)
                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)
            self.images = images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        if self.keep_in_mem:
            img = self.images[index]
        else:
            path = os.path.join(self.root, self.imgs[index])
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.ret_index:
            return img, target
        else:
            return index, img, target

    def __len__(self):
        return len(self.imgs)


# preprocess

means = {"imagenet": [0.485, 0.456, 0.406]}

stds = {"imagenet": [0.229, 0.224, 0.225]}


def get_augmentation(trans_type="aug_0", image_size=224, stat="imagenet"):
    stat = "imagenet"
    mean, std = means[stat], stds[stat]
    image_s = image_size + 32

    data_transforms = {
        "raw": transforms.Compose(
            [
                transforms.Resize((image_s, image_s)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "aug_0": transforms.Compose(
            [
                transforms.Resize((image_s, image_s)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "aug_1": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    return data_transforms[trans_type]


# dataset

datasets_path = {
    "office": "./data/office",
    "office_home": "./data/officehome",
    "visda17": "./data/visda17",
    "domainnet": "./data/domainnet",
}


def create_dataset(
    name,
    domain,
    txt="",
    suffix="",
    keep_in_mem=False,
    ret_index=False,
    image_transform=None,
    use_mean_std=False,
    image_size=224,
):
    if suffix != "":
        suffix = "_" + suffix
    if txt == "":
        txt = f"{domain}{suffix}"

    stat = f"{name}_{domain}" if use_mean_std else "imagenet"
    if image_transform is not None and isinstance(image_transform, str):
        transform = get_augmentation(image_transform, stat=stat, image_size=image_size)

    return Imagelists(
        f"data/splits/{name}/{txt}.txt",
        datasets_path[name],
        keep_in_mem=keep_in_mem,
        ret_index=ret_index,
        transform=transform,
    )


# dataloader


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def worker_init_seed(worker_id):
    np.random.seed(12 + worker_id)
    random.seed(12 + worker_id)


def create_loader(dataset, batch_size=32, num_workers=4, is_train=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        num_workers=num_workers,
        shuffle=is_train,
        drop_last=is_train,
        pin_memory=True,
        worker_init_fn=worker_init_seed,
    )
