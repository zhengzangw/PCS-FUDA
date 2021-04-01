import datetime
import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pcs.utils import print_info, torchutils
from torch.utils.tensorboard import SummaryWriter


class BaseAgent(object):
    """
    General agent class

    Abstract Methods to be implemented:

        _load_datasets
        _create_model
        _create_optimizer
        train_one_epoch
        validate
        load_checkpoint
        save_checkpoint
    """

    def __init__(self, config):
        self.config = config
        # set seed as early as possible
        torchutils.set_seed(self.config.seed)

        self.model = None
        self.optim = None
        self.logger = logging.getLogger("Agent")
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0
        self.val_acc = []
        self.train_loss = []
        self.lr_scheduler_list = []

        print_info(self.logger.info)
        self.starttime = datetime.datetime.now()
        self._choose_device()

        # Load Dataset
        self._load_datasets()

        self._create_model()
        self._create_optimizer()

        # we need these to decide best loss
        self.current_loss = 0.0
        self.current_val_metric = 0.0
        self.best_val_metric = 0.0
        self.best_val_epoch = 0
        self.iter_with_no_improv = 0

    def get_attr(self, domain, name):
        return getattr(self, f"{name}_{domain}")

    def set_attr(self, domain, name, value):
        setattr(self, f"{name}_{domain}", value)
        return self.get_attr(domain, name)

    def _choose_device(self):
        # check if use gpu
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA"
            )
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            cudnn.benchmark = True

            if self.config.gpu_device is None:
                self.config.gpu_device = list(range(torch.cuda.device_count()))
            elif not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]
            self.gpu_devices = self.config.gpu_device

            # set device when only one gpu
            num_gpus = len(self.gpu_devices)
            self.multigpu = num_gpus > 1 and torch.cuda.device_count() > 1
            if not self.multigpu:
                torch.cuda.set_device(self.gpu_devices[0])

            gpu_devices = ",".join([str(_gpu_id) for _gpu_id in self.gpu_devices])
            self.logger.info(f"User specified {num_gpus} GPUs: {gpu_devices}")
            self.parallel_helper_idxs = torch.arange(len(self.gpu_devices)).to(
                self.device
            )

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            torchutils.print_cuda_statistics(output=self.logger.info, nvidia_smi=False)
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
            self.cleanup()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            self.cleanup()
            raise e
        except Exception as e:
            self.logger.error(e, exc_info=True)

    def train(self):
        """
        Main training loop
        :return:
        """
        if self.config.validate_freq:
            self.validate()

        for epoch in range(self.current_epoch + 1, self.config.num_epochs + 1):
            # early stop
            patience = self.config.optim_params.patience
            if patience and self.iter_with_no_improv > patience:
                self.logger.info(
                    f"accuracy not improved in {patience} epoches, stopped"
                )
                break
            # train
            self.current_epoch = epoch
            self.train_one_epoch()
            # validate
            if self.config.validate_freq and epoch % self.config.validate_freq == 0:
                self.validate()
            # adjust
            for sch in self.lr_scheduler_list:
                sch.step()
            # save
            self.save_checkpoint()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.summary_writer.close()
        self.save_checkpoint(filename="backup.pth.tar")

    def finalise(self):
        """
        Do appropriate saving after model is :finished training
        """
        self.backup()

    def cleanup(self):
        """
        Undo any global changes that the Agent may have made
        """
        if hasattr(self, "best_val_epoch"):
            self.logger.info(
                f"Best Val acc at {self.best_val_epoch}: {self.best_val_metric:.3}"
            )
        endtime = datetime.datetime.now()
        exe_time = endtime - self.starttime
        self.logger.info(
            f"End at time: {endtime.strftime('%Y.%m.%d-%H:%M:%S')}, total time: {exe_time.seconds}s"
        )

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if (
            self.config.copy_checkpoint_freq
            and self.current_epoch % self.config.copy_checkpoint_freq == 0
        ):
            self.logger.info(f"Backup checkpoint_epoch_{self.current_epoch}.pth.tar")
            torchutils.copy_checkpoint(
                filename=filename,
                folder=self.config.checkpoint_dir,
                copyname=f"checkpoint_epoch_{self.current_epoch}.pth.tar",
            )

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError
