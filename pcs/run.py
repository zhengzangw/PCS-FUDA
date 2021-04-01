#!/usr/bin/env python3

import argparse
import os

from pcs.agents import *
from pcs.utils import check_pretrain_dir, load_json, process_config, set_default


def adjust_config(config):
    set_default(config, "validate_freq", value=1)
    set_default(config, "copy_checkpoint_freq", value=50)
    set_default(config, "debug", value=False)
    set_default(config, "cuda", value=True)
    set_default(config, "gpu_device", value=None)
    set_default(config, "pretrained_exp_dir", value=None)
    set_default(config, "agent", value="CDSAgent")

    # data_params
    set_default(config.data_params, "aug_src", callback="aug")
    set_default(config.data_params, "aug_tgt", callback="aug")
    set_default(config.data_params, "num_workers", value=4)
    set_default(config.data_params, "image_size", value=224)

    # model_params
    set_default(config.model_params, "load_weight_epoch", value=0)
    set_default(config.model_params, "load_memory_bank", value=True)

    # loss_params
    num_loss = len(config.loss_params.loss)
    set_default(config.loss_params, "weight", value=[1] * num_loss)
    set_default(config.loss_params, "start", value=[0] * num_loss)
    set_default(config.loss_params, "end", value=[1000] * num_loss)
    if not isinstance(config.loss_params.temp, list):
        config.loss_params.temp = [config.loss_params.temp] * num_loss
    assert len(config.loss_params.weight) == num_loss
    set_default(config.loss_params, "m", value=0.5)
    set_default(config.loss_params, "T", value=0.05)
    set_default(config.loss_params, "pseudo", value=True)

    # optim_params
    set_default(config.optim_params, "batch_size_src", callback="batch_size")
    set_default(config.optim_params, "batch_size_tgt", callback="batch_size")
    set_default(config.optim_params, "batch_size_lbd", callback="batch_size")
    set_default(config.optim_params, "momentum", value=0.9)
    set_default(config.optim_params, "nesterov", value=True)
    set_default(config.optim_params, "lr_decay_rate", value=0.1)
    set_default(config.optim_params, "cls_update", value=True)

    # clustering
    if config.loss_params.clus is not None:
        if config.loss_params.clus.type is None:
            config.loss_params.clus = None
        else:
            if not isinstance(config.loss_params.clus.type, list):
                config.loss_params.clus.type = [config.loss_params.clus.type]
            k = config.loss_params.clus.k
            n_k = config.loss_params.clus.n_k
            config.k_list = k * n_k
            config.loss_params.clus.n_kmeans = len(config.k_list)

    return config


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/office.json",
        help="the path to the config",
    )
    parser.add_argument("--exp_id", type=str, default=None)
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["office", "office_home", "visda17"],
        help="the name of dataset",
    )
    parser.add_argument("--source", type=str, default=None, help="source domain")
    parser.add_argument("--target", type=str, default=None, help="target domain")
    parser.add_argument(
        "--num", type=str, default=None, help="number of labeled examples in the target"
    )

    # Model
    parser.add_argument("--net", type=str, default=None, help="which network to use")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["S+T", "ENT", "MME"],
        help="MME is proposed method, ENT is entropy minimization, S+T is training only on labeled examples",
    )

    # Optim
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="maximum number of iterations to train (default: 50000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--multi",
        type=float,
        default=None,
        metavar="MLT",
        help="learning rate multiplication",
    )
    parser.add_argument(
        "--early",
        action="store_false",
        default=True,
        help="early stopping on validation or not",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        metavar="S",
        help="early stopping to wait for improvment "
        "before terminating. (default: 5 (5000 iterations))",
    )
    # Hyper-parameter
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--T", type=float, default=None, metavar="T", help="temperature (default: 0.05)"
    )
    parser.add_argument(
        "--lamda", type=float, default=None, metavar="LAM", help="value of lamda"
    )
    # Save model
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        metavar="N",
        help="how many batches to wait before saving a model",
    )
    parser.add_argument(
        "--save_check", action="store_true", default=None, help="save checkpoint or not"
    )
    parser.add_argument(
        "--checkpath", type=str, default=None, help="dir to save checkpoint"
    )

    # Others
    parser.add_argument("--loop", action="store_true")

    return parser


def update_config(config, args):
    if args.dataset:
        config_json["data_params"]["name"] = args.dataset
    if args.source:
        config_json["data_params"]["source"] = args.source
    if args.target:
        config_json["data_params"]["target"] = args.target
    if args.num:
        config_json["data_params"]["fewshot"] = args.num
    if args.exp_id:
        config_json["exp_id"] = args.exp_id
    elif args.source:
        config_json["exp_id"] = f"{args.source}->{args.target}:{args.num}"
    if args.seed:
        config_json["seed"] = args.seed
    if args.lr:
        config_json["optim_params"]["learning_rate"] = args.lr


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    # load config
    config_json = load_json(args.config)
    update_config(config_json, args)

    # check pretrain directory
    pre_checkpoint_dir = check_pretrain_dir(config_json)

    # json to DotMap
    config = process_config(config_json)
    config = adjust_config(config)

    # create agent
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    if pre_checkpoint_dir is not None:
        agent.load_checkpoint("model_best.pth.tar", pre_checkpoint_dir)
    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass
