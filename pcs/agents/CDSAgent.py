import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pcs.models import (CosineClassifier, MemoryBank, SSDALossModule,
                        compute_variance, loss_info, torch_kmeans,
                        update_data_memory)
from pcs.utils import (AverageMeter, datautils, is_div, per, reverse_domain,
                       torchutils, utils)
from sklearn import metrics
from tqdm import tqdm

from . import BaseAgent

ls_abbr = {
    "cls-so": "cls",
    "proto-each": "P",
    "proto-src": "Ps",
    "proto-tgt": "Pt",
    "cls-info": "info",
    "I2C-cross": "C",
    "semi-condentmax": "sCE",
    "semi-entmin": "sE",
    "tgt-condentmax": "tCE",
    "tgt-entmin": "tE",
    "ID-each": "I",
    "CD-cross": "CD",
}


class CDSAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self._define_task(config)
        self.is_features_computed = False
        self.current_iteration_source = self.current_iteration_target = 0
        self.domain_map = {
            "source": self.config.data_params.source,
            "target": self.config.data_params.target,
        }

        super(CDSAgent, self).__init__(config)

        # for MIM
        self.momentum_softmax_target = torchutils.MomentumSoftmax(
            self.num_class, m=len(self.get_attr("target", "train_loader"))
        )
        self.momentum_softmax_source = torchutils.MomentumSoftmax(
            self.num_class, m=len(self.get_attr("source", "train_loader"))
        )

        # init loss
        loss_fn = SSDALossModule(self.config, gpu_devices=self.gpu_devices)
        loss_fn = nn.DataParallel(loss_fn, device_ids=self.gpu_devices).cuda()
        self.loss_fn = loss_fn

        if self.config.pretrained_exp_dir is None:
            self._init_memory_bank()

        # init statics
        self._init_labels()
        self._load_fewshot_to_cls_weight()

    def _define_task(self, config):
        # specify task
        self.fewshot = config.data_params.fewshot
        self.clus = config.loss_params.clus != None
        self.cls = self.semi = self.tgt = self.ssl = False
        self.is_pseudo_src = self.is_pseudo_tgt = False
        for ls in config.loss_params.loss:
            self.cls = self.cls | ls.startswith("cls")
            self.semi = self.semi | ls.startswith("semi")
            self.tgt = self.tgt | ls.startswith("tgt")
            self.ssl = self.ssl | (ls.split("-")[0] not in ["cls", "semi", "tgt"])
            self.is_pseudo_src = self.is_pseudo_src | ls.startswith("semi-pseudo")
            self.is_pseudo_tgt = self.is_pseudo_tgt | ls.startswith("tgt-pseudo")

        self.is_pseudo_src = self.is_pseudo_src | (
            config.loss_params.pseudo and self.fewshot is not None
        )
        self.is_pseudo_tgt = self.is_pseudo_tgt | config.loss_params.pseudo
        self.semi = self.semi | self.is_pseudo_src
        if self.clus:
            self.is_pseudo_tgt = self.is_pseudo_tgt | (
                config.loss_params.clus.tgt_GC == "PGC" and "GC" in config.clus.type
            )

    def _init_labels(self):
        train_len_tgt = self.get_attr("target", "train_len")
        train_len_src = self.get_attr("source", "train_len")

        # labels for pseudo
        if self.fewshot:
            self.predict_ordered_labels_pseudo_source = (
                torch.zeros(train_len_src, dtype=torch.long).detach().cuda() - 1
            )
            for ind, lbl in zip(self.fewshot_index_source, self.fewshot_label_source):
                self.predict_ordered_labels_pseudo_source[ind] = lbl
        self.predict_ordered_labels_pseudo_target = (
            torch.zeros(train_len_tgt, dtype=torch.long).detach().cuda() - 1
        )

    def _load_datasets(self):
        name = self.config.data_params.name
        num_workers = self.config.data_params.num_workers
        fewshot = self.config.data_params.fewshot
        domain = self.domain_map
        image_size = self.config.data_params.image_size
        aug_src = self.config.data_params.aug_src
        aug_tgt = self.config.data_params.aug_tgt
        raw = "raw"

        self.num_class = datautils.get_class_num(
            f'data/splits/{name}/{domain["source"]}.txt'
        )
        self.class_map = datautils.get_class_map(
            f'data/splits/{name}/{domain["target"]}.txt'
        )

        batch_size_dict = {
            "test": self.config.optim_params.batch_size,
            "source": self.config.optim_params.batch_size_src,
            "target": self.config.optim_params.batch_size_tgt,
            "labeled": self.config.optim_params.batch_size_lbd,
        }
        self.batch_size_dict = batch_size_dict

        # self-supervised Dataset
        for domain_name in ("source", "target"):
            aug_name = {"source": aug_src, "target": aug_tgt}[domain_name]

            # Training datasets
            train_dataset = datautils.create_dataset(
                name,
                domain[domain_name],
                suffix="",
                ret_index=True,
                image_transform=aug_name,
                use_mean_std=False,
                image_size=image_size,
            )

            train_loader = datautils.create_loader(
                train_dataset,
                batch_size_dict[domain_name],
                is_train=True,
                num_workers=num_workers,
            )
            train_init_loader = datautils.create_loader(
                train_dataset,
                batch_size_dict[domain_name],
                is_train=False,
                num_workers=num_workers,
            )
            train_labels = torch.from_numpy(train_dataset.labels).detach().cuda()

            self.set_attr(domain_name, "train_dataset", train_dataset)
            self.set_attr(domain_name, "train_ordered_labels", train_labels)
            self.set_attr(domain_name, "train_loader", train_loader)
            self.set_attr(domain_name, "train_init_loader", train_init_loader)
            self.set_attr(domain_name, "train_len", len(train_dataset))

        # Classification and Fewshot Dataset

        if fewshot:
            train_lbd_dataset_source = datautils.create_dataset(
                name,
                domain["source"],
                suffix=f"labeled_{fewshot}",
                ret_index=True,
                image_transform=aug_src,
                image_size=image_size,
            )
            src_dataset = self.get_attr("source", "train_dataset")
            (
                self.fewshot_index_source,
                self.fewshot_label_source,
            ) = datautils.get_fewshot_index(train_lbd_dataset_source, src_dataset)

            test_unl_dataset_source = datautils.create_dataset(
                name,
                domain["source"],
                suffix=f"unlabeled_{fewshot}",
                ret_index=True,
                image_transform=raw,
                image_size=image_size,
            )
            self.test_unl_loader_source = datautils.create_loader(
                test_unl_dataset_source,
                batch_size_dict["test"],
                is_train=False,
                num_workers=num_workers,
            )

            # labels for fewshot
            train_len = self.get_attr("source", "train_len")
            self.fewshot_labels = (
                torch.zeros(train_len, dtype=torch.long).detach().cuda() - 1
            )
            for ind, lbl in zip(self.fewshot_index_source, self.fewshot_label_source):
                self.fewshot_labels[ind] = lbl

        else:
            train_lbd_dataset_source = datautils.create_dataset(
                name,
                domain["source"],
                ret_index=True,
                image_transform=aug_src,
                image_size=image_size,
            )

        test_suffix = "test" if self.config.data_params.train_val_split else ""
        test_unl_dataset_target = datautils.create_dataset(
            name,
            domain["target"],
            suffix=test_suffix,
            ret_index=True,
            image_transform=raw,
            image_size=image_size,
        )

        self.train_lbd_loader_source = datautils.create_loader(
            train_lbd_dataset_source,
            batch_size_dict["labeled"],
            num_workers=num_workers,
        )
        self.test_unl_loader_target = datautils.create_loader(
            test_unl_dataset_target,
            batch_size_dict["test"],
            is_train=False,
            num_workers=num_workers,
        )

        self.logger.info(
            f"Dataset {name}, source {self.config.data_params.source}, target {self.config.data_params.target}"
        )

    def _create_model(self):
        version_grp = self.config.model_params.version.split("-")
        version = version_grp[-1]
        pretrained = "pretrain" in version_grp
        if pretrained:
            self.logger.info("Imagenet pretrained model used")
        out_dim = self.config.model_params.out_dim

        # backbone
        if "resnet" in version:
            net_class = getattr(torchvision.models, version)

            if pretrained:
                model = net_class(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, out_dim)
                torchutils.weights_init(model.fc)
            else:
                model = net_class(pretrained=False, num_classes=out_dim)
        else:
            raise NotImplementedError

        model = nn.DataParallel(model, device_ids=self.gpu_devices)
        model = model.cuda()
        self.model = model

        # classification head
        if self.cls:
            self.criterion = nn.CrossEntropyLoss().cuda()
            cls_head = CosineClassifier(
                num_class=self.num_class, inc=out_dim, temp=self.config.loss_params.T
            )
            torchutils.weights_init(cls_head)
            self.cls_head = cls_head.cuda()

    def _create_optimizer(self):
        lr = self.config.optim_params.learning_rate
        momentum = self.config.optim_params.momentum
        weight_decay = self.config.optim_params.weight_decay
        conv_lr_ratio = self.config.optim_params.conv_lr_ratio

        parameters = []
        # batch_norm layer: no weight_decay
        params_bn, _ = torchutils.split_params_by_name(self.model, "bn")
        parameters.append({"params": params_bn, "weight_decay": 0.0})
        # conv layer: small lr
        _, params_conv = torchutils.split_params_by_name(self.model, ["fc", "bn"])
        if conv_lr_ratio:
            parameters[0]["lr"] = lr * conv_lr_ratio
            parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
        else:
            parameters.append({"params": params_conv})
        # fc layer
        params_fc, _ = torchutils.split_params_by_name(self.model, "fc")
        if self.cls and self.config.optim_params.cls_update:
            params_fc.extend(list(self.cls_head.parameters()))
        parameters.append({"params": params_fc})

        self.optim = torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=self.config.optim_params.nesterov,
        )

        # lr schedular
        if self.config.optim_params.lr_decay_schedule:
            optim_stepLR = torch.optim.lr_scheduler.MultiStepLR(
                self.optim,
                milestones=self.config.optim_params.lr_decay_schedule,
                gamma=self.config.optim_params.lr_decay_rate,
            )
            self.lr_scheduler_list.append(optim_stepLR)

        if self.config.optim_params.decay:
            self.optim_iterdecayLR = torchutils.lr_scheduler_invLR(self.optim)

    def train_one_epoch(self):
        # train preparation
        self.model = self.model.train()
        if self.cls:
            self.cls_head.train()
        self.loss_fn.module.epoch = self.current_epoch

        loss_list = self.config.loss_params.loss
        loss_weight = self.config.loss_params.weight
        loss_warmup = self.config.loss_params.start
        loss_giveup = self.config.loss_params.end

        num_loss = len(loss_list)

        source_loader = self.get_attr("source", "train_loader")
        target_loader = self.get_attr("target", "train_loader")
        if self.config.steps_epoch is None:
            num_batches = max(len(source_loader), len(target_loader)) + 1
            self.logger.info(f"source loader batches: {len(source_loader)}")
            self.logger.info(f"target loader batches: {len(target_loader)}")
        else:
            num_batches = self.config.steps_epoch

        epoch_loss = AverageMeter()
        epoch_loss_parts = [AverageMeter() for _ in range(num_loss)]

        # cluster
        if self.clus:
            if self.config.loss_params.clus.kmeans_freq:
                kmeans_batches = num_batches // self.config.loss_params.clus.kmeans_freq
            else:
                kmeans_batches = 1
        else:
            kmeans_batches = None

        # load weight
        self._load_fewshot_to_cls_weight()
        if self.fewshot:
            fewshot_index = torch.tensor(self.fewshot_index_source).cuda()

        tqdm_batch = tqdm(
            total=num_batches, desc=f"[Epoch {self.current_epoch}]", leave=False
        )
        tqdm_post = {}
        for batch_i in range(num_batches):
            # Kmeans
            if is_div(kmeans_batches, batch_i):
                self._update_cluster_labels()

            if not self.config.optim_params.cls_update:
                self._load_fewshot_to_cls_weight()

            # iteration over all source images
            if not batch_i % len(source_loader):
                source_iter = iter(source_loader)

                if "semi-condentmax" in self.config.loss_params.loss:
                    momentum_prob_source = (
                        self.momentum_softmax_source.softmax_vector.cuda()
                    )
                    self.momentum_softmax_source.reset()

            # iteration over all target images
            if not batch_i % len(target_loader):
                target_iter = iter(target_loader)

                if "tgt-condentmax" in self.config.loss_params.loss:
                    momentum_prob_target = (
                        self.momentum_softmax_target.softmax_vector.cuda()
                    )
                    self.momentum_softmax_target.reset()

            # iteration over all labeled source images
            if self.cls and not batch_i % len(self.train_lbd_loader_source):
                source_lbd_iter = iter(self.train_lbd_loader_source)

            # calculate loss
            for domain_name in ("source", "target"):
                loss = torch.tensor(0).cuda()
                loss_d = 0
                loss_part_d = [0] * num_loss
                batch_size = self.batch_size_dict[domain_name]

                if self.cls and domain_name == "source":
                    indices_lbd, images_lbd, labels_lbd = next(source_lbd_iter)
                    indices_lbl = indices_lbd.cuda()
                    images_lbd = images_lbd.cuda()
                    labels_lbd = labels_lbd.cuda()
                    feat_lbd = self.model(images_lbd)
                    feat_lbd = F.normalize(feat_lbd, dim=1)
                    out_lbd = self.cls_head(feat_lbd)

                # Matching & ssl
                if (self.tgt and domain_name == "target") or self.ssl:
                    loader_iter = (
                        source_iter if domain_name == "source" else target_iter
                    )

                    indices_unl, images_unl, _ = next(loader_iter)
                    images_unl = images_unl.cuda()
                    indices_unl = indices_unl.cuda()
                    feat_unl = self.model(images_unl)
                    feat_unl = F.normalize(feat_unl, dim=1)
                    out_unl = self.cls_head(feat_unl)

                # Semi Supervised
                if self.semi and domain_name == "source":
                    semi_mask = ~torchutils.isin(indices_unl, fewshot_index)

                    indices_semi = indices_unl[semi_mask]
                    out_semi = out_unl[semi_mask]

                # Self-supervised Learning
                if self.ssl:
                    _, new_data_memory, loss_ssl, aux_list = self.loss_fn(
                        indices_unl, feat_unl, domain_name, self.parallel_helper_idxs
                    )
                    loss_ssl = [torch.mean(ls) for ls in loss_ssl]

                # pseudo
                loss_pseudo = torch.tensor(0).cuda()
                is_pseudo = {"source": self.is_pseudo_src, "target": self.is_pseudo_tgt}
                thres_dict = {
                    "source": self.config.loss_params.thres_src,
                    "target": self.config.loss_params.thres_tgt,
                }

                if is_pseudo[domain_name]:
                    if domain_name == "source":
                        indices_pseudo = indices_semi
                        out_pseudo = out_semi
                        pseudo_domain = self.predict_ordered_labels_pseudo_source
                    else:
                        indices_pseudo = indices_unl
                        out_pseudo = out_unl  # [bs, class_num]
                        pseudo_domain = self.predict_ordered_labels_pseudo_target
                    thres = thres_dict[domain_name]

                    # calculate loss
                    loss_pseudo, aux = torchutils.pseudo_label_loss(
                        out_pseudo,
                        thres=thres,
                        mask=None,
                        num_class=self.num_class,
                        aux=True,
                    )
                    mask_pseudo = aux["mask"]

                    # fewshot memory bank
                    mb = self.get_attr("source", "memory_bank_wrapper")
                    indices_lbd_tounl = fewshot_index[indices_lbd]
                    mb_feat_lbd = mb.at_idxs(indices_lbd_tounl)
                    fewshot_data_memory = update_data_memory(mb_feat_lbd, feat_lbd)

                    # stat
                    pred_selected = out_pseudo.argmax(dim=1)[mask_pseudo]
                    indices_selected = indices_pseudo[mask_pseudo]
                    indices_unselected = indices_pseudo[~mask_pseudo]

                    pseudo_domain[indices_selected] = pred_selected
                    pseudo_domain[indices_unselected] = -1

                # Compute Loss

                for ind, ls in enumerate(loss_list):
                    if (
                        self.current_epoch < loss_warmup[ind]
                        or self.current_epoch >= loss_giveup[ind]
                    ):
                        continue
                    loss_part = torch.tensor(0).cuda()
                    # *** handler for different loss ***
                    # classification on few-shot
                    if ls == "cls-so" and domain_name == "source":
                        loss_part = self.criterion(out_lbd, labels_lbd)
                    elif ls == "cls-info" and domain_name == "source":
                        loss_part = loss_info(feat_lbd, mb_feat_lbd, labels_lbd)
                    # semi-supervision learning on unlabled source
                    elif ls == "semi-entmin" and domain_name == "source":
                        loss_part = torchutils.entropy(out_semi)
                    elif ls == "semi-condentmax" and domain_name == "source":
                        bs = out_semi.size(0)
                        prob_semi = F.softmax(out_semi, dim=1)
                        prob_mean_semi = prob_semi.sum(dim=0) / bs

                        # update momentum
                        self.momentum_softmax_source.update(
                            prob_mean_semi.cpu().detach(), bs
                        )
                        # get momentum probability
                        momentum_prob_source = (
                            self.momentum_softmax_source.softmax_vector.cuda()
                        )
                        # compute loss
                        entropy_cond = -torch.sum(
                            prob_mean_semi * torch.log(momentum_prob_source + 1e-5)
                        )
                        loss_part = -entropy_cond

                    # learning on unlabeled target domain
                    elif ls == "tgt-entmin" and domain_name == "target":
                        loss_part = torchutils.entropy(out_unl)
                    elif ls == "tgt-condentmax" and domain_name == "target":
                        bs = out_unl.size(0)
                        prob_unl = F.softmax(out_unl, dim=1)
                        prob_mean_unl = prob_unl.sum(dim=0) / bs

                        # update momentum
                        self.momentum_softmax_target.update(
                            prob_mean_unl.cpu().detach(), bs
                        )
                        # get momentum probability
                        momentum_prob_target = (
                            self.momentum_softmax_target.softmax_vector.cuda()
                        )
                        # compute loss
                        entropy_cond = -torch.sum(
                            prob_mean_unl * torch.log(momentum_prob_target + 1e-5)
                        )
                        loss_part = -entropy_cond
                    # self-supervised learning
                    elif ls.split("-")[0] in ["ID", "CD", "proto", "I2C", "C2C"]:
                        loss_part = loss_ssl[ind]

                    loss_part = loss_weight[ind] * loss_part
                    loss = loss + loss_part
                    loss_d = loss_d + loss_part.item()
                    loss_part_d[ind] = loss_part.item()

                # Backpropagation
                self.optim.zero_grad()
                if len(loss_list) and loss != 0:
                    loss.backward()
                self.optim.step()

                # update memory_bank
                if self.ssl:
                    self._update_memory_bank(domain_name, indices_unl, new_data_memory)
                    if domain_name == "source":
                        self._update_memory_bank(
                            domain_name, indices_lbd_tounl, fewshot_data_memory
                        )

                # update lr info
                tqdm_post["lr"] = torchutils.get_lr(self.optim, g_id=-1)

                # update loss info
                epoch_loss.update(loss_d, batch_size)
                tqdm_post["loss"] = epoch_loss.avg
                self.summary_writer.add_scalars(
                    "train/loss", {"loss": epoch_loss.val}, self.current_iteration
                )
                self.train_loss.append(epoch_loss.val)

                # update loss part info
                domain_iteration = self.get_attr(domain_name, "current_iteration")
                self.summary_writer.add_scalars(
                    f"train/{self.domain_map[domain_name]}_loss",
                    {"loss": epoch_loss.val},
                    domain_iteration,
                )
                for i, ls in enumerate(loss_part_d):
                    ls_name = loss_list[i]
                    epoch_loss_parts[i].update(ls, batch_size)
                    tqdm_post[ls_abbr[ls_name]] = epoch_loss_parts[i].avg
                    self.summary_writer.add_scalars(
                        f"train/{self.domain_map[domain_name]}_loss",
                        {ls_name: epoch_loss_parts[i].val},
                        domain_iteration,
                    )

                # adjust lr
                if self.config.optim_params.decay:
                    self.optim_iterdecayLR.step()

                self.current_iteration += 1
            tqdm_batch.set_postfix(tqdm_post)
            tqdm_batch.update()
            self.current_iteration_source += 1
            self.current_iteration_target += 1
        tqdm_batch.close()

        self.current_loss = epoch_loss.avg

    @torch.no_grad()
    def _load_fewshot_to_cls_weight(self):
        """load centroids to cosine classifier

        Args:
            method (str, optional): None, 'fewshot', 'src', 'tgt'. Defaults to None.
        """
        method = self.config.model_params.load_weight

        if method is None:
            return
        assert method in ["fewshot", "src", "tgt", "src-tgt", "fewshot-tgt"]

        thres = {"src": 1, "tgt": self.config.model_params.load_weight_thres}
        bank = {
            "src": self.get_attr("source", "memory_bank_wrapper").as_tensor(),
            "tgt": self.get_attr("target", "memory_bank_wrapper").as_tensor(),
        }
        fewshot_label = {}
        fewshot_index = {}
        is_tgt = (
            method in ["tgt", "fewshot-tgt", "src-tgt"]
            and self.current_epoch >= self.config.model_params.load_weight_epoch
        )
        if method in ["fewshot", "fewshot-tgt"]:
            if self.fewshot:
                fewshot_label["src"] = torch.tensor(self.fewshot_label_source)
                fewshot_index["src"] = torch.tensor(self.fewshot_index_source)
            else:
                fewshot_label["src"] = self.get_attr("source", "train_ordered_labels")
                fewshot_index["src"] = torch.arange(
                    self.get_attr("source", "train_len")
                )

        else:
            mask = self.predict_ordered_labels_pseudo_source != -1
            fewshot_label["src"] = self.predict_ordered_labels_pseudo_source[mask]
            fewshot_index["src"] = mask.nonzero().squeeze(1)
        if is_tgt:
            mask = self.predict_ordered_labels_pseudo_target != -1
            fewshot_label["tgt"] = self.predict_ordered_labels_pseudo_target[mask]
            fewshot_index["tgt"] = mask.nonzero().squeeze(1)

        for domain in ("src", "tgt"):
            if domain == "tgt" and not is_tgt:
                break
            if domain == "src" and method == "tgt":
                break
            weight = self.cls_head.fc.weight.data

            for label in range(self.num_class):
                fewshot_mask = fewshot_label[domain] == label
                if fewshot_mask.sum() < thres[domain]:
                    continue
                fewshot_ind = fewshot_index[domain][fewshot_mask]
                bank_vec = bank[domain][fewshot_ind]
                weight[label] = F.normalize(torch.mean(bank_vec, dim=0), dim=0)

    # Validate

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        # Domain Adaptation
        if self.cls:
            # self._load_fewshot_to_cls_weight()
            self.cls_head.eval()
            if (
                self.config.data_params.fewshot
                and self.config.data_params.name not in ["visda17", "digits"]
            ):
                self.score(
                    self.test_unl_loader_source,
                    name=f"unlabeled {self.domain_map['source']}",
                )
            self.current_val_metric = self.score(
                self.test_unl_loader_target,
                name=f"unlabeled {self.domain_map['target']}",
            )

        # update information
        self.current_val_iteration += 1
        if self.current_val_metric >= self.best_val_metric:
            self.best_val_metric = self.current_val_metric
            self.best_val_epoch = self.current_epoch
            self.iter_with_no_improv = 0
        else:
            self.iter_with_no_improv += 1
        self.val_acc.append(self.current_val_metric)

        self.clear_train_features()

    @torch.no_grad()
    def score(self, loader, name="test"):
        correct = 0
        size = 0
        epoch_loss = AverageMeter()
        error_indices = []
        confusion_matrix = torch.zeros(self.num_class, self.num_class, dtype=torch.long)
        pred_score = []
        pred_label = []
        label = []

        for batch_i, (indices, images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            feat = self.model(images)
            feat = F.normalize(feat, dim=1)
            output = self.cls_head(feat)
            prob = F.softmax(output, dim=-1)

            loss = self.criterion(output, labels)
            pred = torch.max(output, dim=1)[1]

            pred_label.extend(pred.cpu().tolist())
            label.extend(labels.cpu().tolist())
            if self.num_class == 2:
                pred_score.extend(prob[:, 1].cpu().tolist())

            correct += pred.eq(labels).sum().item()
            for t, p, ind in zip(labels, pred, indices):
                confusion_matrix[t.long(), p.long()] += 1
                if t != p:
                    error_indices.append((ind, p))
            size += pred.size(0)
            epoch_loss.update(loss, pred.size(0))

        acc = correct / size
        self.summary_writer.add_scalars(
            "test/acc", {f"{name}": acc}, self.current_epoch
        )
        self.summary_writer.add_scalars(
            "test/loss", {f"{name}": epoch_loss.avg}, self.current_epoch
        )
        self.logger.info(
            f"[Epoch {self.current_epoch} {name}] loss={epoch_loss.avg:.5f}, acc={correct}/{size}({100. * acc:.3f}%)"
        )

        return acc

    # Load & Save checkpoint

    def load_checkpoint(
        self,
        filename,
        checkpoint_dir=None,
        load_memory_bank=False,
        load_model=True,
        load_optim=False,
        load_epoch=False,
        load_cls=True,
    ):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location="cpu")

            if load_epoch:
                self.current_epoch = checkpoint["epoch"]
                for domain_name in ("source", "target"):
                    self.set_attr(
                        domain_name,
                        "current_iteration",
                        checkpoint[f"iteration_{domain_name}"],
                    )
                self.current_iteration = checkpoint["iteration"]
                self.current_val_iteration = checkpoint["val_iteration"]

            if load_model:
                model_state_dict = checkpoint["model_state_dict"]
                self.model.load_state_dict(model_state_dict)

            if load_cls and self.cls and "cls_state_dict" in checkpoint:
                cls_state_dict = checkpoint["cls_state_dict"]
                self.cls_head.load_state_dict(cls_state_dict)

            if load_optim:
                optim_state_dict = checkpoint["optim_state_dict"]
                self.optim.load_state_dict(optim_state_dict)

                lr_pretrained = self.optim.param_groups[0]["lr"]
                lr_config = self.config.optim_params.learning_rate

                # Change learning rate
                if not lr_pretrained == lr_config:
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = self.config.optim_params.learning_rate

            self._init_memory_bank()
            if (
                load_memory_bank or self.config.model_params.load_memory_bank == False
            ):  # load memory_bank
                self._load_memory_bank(
                    {
                        "source": checkpoint["memory_bank_source"],
                        "target": checkpoint["memory_bank_target"],
                    }
                )

            self.logger.info(
                f"Checkpoint loaded successfully from '{filename}' at (epoch {checkpoint['epoch']}) at (iteration s:{checkpoint['iteration_source']} t:{checkpoint['iteration_target']}) with loss = {checkpoint['loss']}\nval acc = {checkpoint['val_acc']}\n"
            )

        except OSError as e:
            self.logger.info(f"Checkpoint doesnt exists: [{filename}]")
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "memory_bank_source": self.get_attr("source", "memory_bank_wrapper"),
            "memory_bank_target": self.get_attr("target", "memory_bank_wrapper"),
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "iteration_source": self.get_attr("source", "current_iteration"),
            "iteration_target": self.get_attr("target", "current_iteration"),
            "val_iteration": self.current_val_iteration,
            "val_acc": np.array(self.val_acc),
            "val_metric": self.current_val_metric,
            "loss": self.current_loss,
            "train_loss": np.array(self.train_loss),
        }
        if self.cls:
            out_dict["cls_state_dict"] = self.cls_head.state_dict()
        # best according to source-to-target
        is_best = (
            self.current_val_metric == self.best_val_metric
        ) or not self.config.validate_freq
        torchutils.save_checkpoint(
            out_dict, is_best, filename=filename, folder=self.config.checkpoint_dir
        )
        self.copy_checkpoint()

    # compute train features

    @torch.no_grad()
    def compute_train_features(self):
        if self.is_features_computed:
            return
        else:
            self.is_features_computed = True
        self.model.eval()

        for domain in ("source", "target"):
            train_loader = self.get_attr(domain, "train_init_loader")
            features, y, idx = [], [], []
            tqdm_batch = tqdm(
                total=len(train_loader), desc=f"[Compute train features of {domain}]"
            )
            for batch_i, (indices, images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                feat = self.model(images)
                feat = F.normalize(feat, dim=1)

                features.append(feat)
                y.append(labels)
                idx.append(indices)

                tqdm_batch.update()
            tqdm_batch.close()

            features = torch.cat(features)
            y = torch.cat(y)
            idx = torch.cat(idx).to(self.device)

            self.set_attr(domain, "train_features", features)
            self.set_attr(domain, "train_labels", y)
            self.set_attr(domain, "train_indices", idx)

    def clear_train_features(self):
        self.is_features_computed = False

    # Memory bank

    @torch.no_grad()
    def _init_memory_bank(self):
        out_dim = self.config.model_params.out_dim
        for domain_name in ("source", "target"):
            data_len = self.get_attr(domain_name, "train_len")
            memory_bank = MemoryBank(data_len, out_dim)
            if self.config.model_params.load_memory_bank:
                self.compute_train_features()
                idx = self.get_attr(domain_name, "train_indices")
                feat = self.get_attr(domain_name, "train_features")
                memory_bank.update(idx, feat)
                # self.logger.info(
                #     f"Initialize memorybank-{domain_name} with pretrained output features"
                # )
                # save space
                if self.config.data_params.name in ["visda17", "domainnet"]:
                    delattr(self, f"train_indices_{domain_name}")
                    delattr(self, f"train_features_{domain_name}")

            self.set_attr(domain_name, "memory_bank_wrapper", memory_bank)

            self.loss_fn.module.set_attr(domain_name, "data_len", data_len)
            self.loss_fn.module.set_broadcast(
                domain_name, "memory_bank", memory_bank.as_tensor()
            )

    @torch.no_grad()
    def _update_memory_bank(self, domain_name, indices, new_data_memory):
        memory_bank_wrapper = self.get_attr(domain_name, "memory_bank_wrapper")
        memory_bank_wrapper.update(indices, new_data_memory)
        updated_bank = memory_bank_wrapper.as_tensor()
        self.loss_fn.module.set_broadcast(domain_name, "memory_bank", updated_bank)

    def _load_memory_bank(self, memory_bank_dict):
        """load memory bank from checkpoint

        Args:
            memory_bank_dict (dict): memory_bank dict of source and target domain
        """
        for domain_name in ("source", "target"):
            memory_bank = memory_bank_dict[domain_name]._bank.cuda()
            self.get_attr(domain_name, "memory_bank_wrapper")._bank = memory_bank
            self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank)

    # Cluster

    @torch.no_grad()
    def _update_cluster_labels(self):
        k_list = self.config.k_list
        for clus_type in self.config.loss_params.clus.type:
            cluster_labels_domain = {}
            cluster_centroids_domain = {}
            cluster_phi_domain = {}

            # clustering for each domain
            if clus_type == "each":
                for domain_name in ("source", "target"):

                    memory_bank_tensor = self.get_attr(
                        domain_name, "memory_bank_wrapper"
                    ).as_tensor()

                    # clustering
                    cluster_labels, cluster_centroids, cluster_phi = torch_kmeans(
                        k_list,
                        memory_bank_tensor,
                        seed=self.current_epoch + self.current_iteration,
                    )

                    cluster_labels_domain[domain_name] = cluster_labels
                    cluster_centroids_domain[domain_name] = cluster_centroids
                    cluster_phi_domain[domain_name] = cluster_phi

                self.cluster_each_centroids_domain = cluster_centroids_domain
                self.cluster_each_labels_domain = cluster_labels_domain
                self.cluster_each_phi_domain = cluster_phi_domain
            else:
                print(clus_type)
                raise NotImplementedError

            # update cluster to losss_fn
            for domain_name in ("source", "target"):
                self.loss_fn.module.set_broadcast(
                    domain_name,
                    f"cluster_labels_{clus_type}",
                    cluster_labels_domain[domain_name],
                )
                self.loss_fn.module.set_broadcast(
                    domain_name,
                    f"cluster_centroids_{clus_type}",
                    cluster_centroids_domain[domain_name],
                )
                if cluster_phi_domain:
                    self.loss_fn.module.set_broadcast(
                        domain_name,
                        f"cluster_phi_{clus_type}",
                        cluster_phi_domain[domain_name],
                    )
