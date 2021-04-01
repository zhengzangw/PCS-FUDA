import datetime
import logging
import os
import shutil
import socket
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint
from tempfile import mkstemp

from dotmap import DotMap

from .utils import load_json, makedirs, save_json


def check_pretrain_dir(config_json):
    pre_checkpoint_dir = None
    if (
        "pretrained_exp_dir" in config_json
        and 
        config_json["pretrained_exp_dir"] is not None
    ):
        print("NOTE: found pretrained model...continue training")
        pre_checkpoint_dir = os.path.join(
            config_json["pretrained_exp_dir"], "checkpoints"
        )
    return pre_checkpoint_dir


def process_config_path(config_path, override_dotmap=None):
    config_json = load_json(config_path)
    return process_config(config_json, override_dotmap=override_dotmap)


def process_config(config_json, override_dotmap=None):
    """
    Processes config file:
        1) Converts it to a DotMap
        2) Creates experiments path and required subdirs
        3) Set up logging
    """
    config = DotMap(config_json)
    if override_dotmap is not None:
        config.update(override_dotmap)

    print("Configuration Loaded:")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    # if config.pretrained_exp_dir is not None:
    #     # don't make new dir more continuing training
    #     exp_dir = config.pretrained_exp_dir
    #     print("[INFO]: Continuing from previously finished training at %s." % exp_dir)
    # else:
    exp_base = config.exp_base

    if config.debug:
        exp_dir = os.path.join(exp_base, "experiments", config.exp_name, "debug")
    else:
        if config.pretrained_exp_dir is not None and isinstance(
            config.pretrained_exp_dir, str
        ):
            # don't make new dir more continuing training
            exp_dir = config.pretrained_exp_dir
            print("[INFO]: Backup previously trained model and config json")
            os.system("cp %s/config.json %s/prev_config.json" % (exp_dir, exp_dir))
            os.system(
                "cp %s/checkpoints/checkpoint.pth.tar %s/checkpoints/prev_checkpoint.pth.tar"
                % (exp_dir, exp_dir)
            )
            os.system(
                "cp %s/checkpoints/model_best.pth.tar %s/checkpoints/prev_model_best.pth.tar"
                % (exp_dir, exp_dir)
            )
        elif config.continue_exp_dir is not None and isinstance(
            config.continue_exp_dir, str
        ):
            exp_dir = config.continue_exp_dir
            print("[INFO]: Backup previously trained model and config json")
            os.system("cp %s/config.json %s/prev_config.json" % (exp_dir, exp_dir))
            os.system(
                "cp %s/checkpoints/checkpoint.pth.tar %s/checkpoints/prev_checkpoint.pth.tar"
                % (exp_dir, exp_dir)
            )
            os.system(
                "cp %s/checkpoints/model_best.pth.tar %s/checkpoints/prev_model_best.pth.tar"
                % (exp_dir, exp_dir)
            )
        else:
            if config.exp_id is None:
                config.exp_id = datetime.datetime.now().strftime("%Y-%m-%d")
            exp_dir = os.path.join(
                exp_base, "experiments", config.exp_name, config.exp_id
            )
            if os.path.exists(exp_dir):
                config.exp_id += "-" + datetime.datetime.now().strftime("%y%m%d%H%M%S")
                exp_dir = os.path.join(
                    exp_base, "experiments", config.exp_name, config.exp_id
                )

    # create some important directories to be used for the experiment.
    config.summary_dir = os.path.join(exp_dir, "summaries/")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    config.out_dir = os.path.join(exp_dir, "out/")
    config.log_dir = os.path.join(exp_dir, "logs/")

    makedirs(
        [config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir]
    )

    # save config to experiment dir
    config_out = os.path.join(exp_dir, "config.json")
    save_json(config.toDict(), config_out)

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Experiment directory is located at %s" % exp_dir)

    logging.getLogger().info("Configurations and directories successfully set up.")
    return config


def setup_logging(log_dir):
    log_file_format = (
        "[%(levelname)s] %(asctime)s: %(message)s in %(pathname)s:%(lineno)d"
    )
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        "{}exp_debug.log".format(log_dir), maxBytes=10 ** 6, backupCount=5
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        "{}exp_error.log".format(log_dir), maxBytes=10 ** 6, backupCount=5
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def print_info(output=print):
    output(f"Start at time: {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")
    output(f"Server: {socket.gethostname()}")


def prepare_dirs(config):
    exp_dir = os.path.join(
        config.exp_base, "experiments", config.exp_name, config.exp_id
    )
    summary_dir = os.path.join(exp_dir, "summaries/")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    out_dir = os.path.join(exp_dir, "out/")
    log_dir = os.path.join(exp_dir, "logs/")
    config.log_file = os.path.join(log_dir, "output.log")
    if config.pretrained_exp_dir == None or config.copy_exp_dir is False:
        makedirs([summary_dir, checkpoint_dir, out_dir, log_dir])
        print(f"Create {exp_dir}")
    else:
        shutil.copytree(config.pretrained_exp_dir, exp_dir)
        if os.path.exists(config.log_file):
            shutil.copy(config.log_file, os.path.join(log_dir, "output_prev.log"))
        print(f"Copy {config.pretrained_exp_dir} to {exp_dir}")
        config.pretrained_exp_dir = exp_dir


def get_cmd(
    config, script_path="/rscratch/xyyue/anaconda3/envs/ssda2/bin/python ./run.py"
):
    config_out = mkstemp()[1]
    save_json(config.toDict(), config_out)
    return f"{script_path} --config {config_out}"
