import os
import subprocess

import wandb.util

from configs.configs_unet import configs as configs_unet
from configs.configs_fcn import configs as configs_fcn

all_configs = {}
all_configs.update(configs_unet)
all_configs.update(configs_fcn)


def load_config(config_id, command, trial_id=None, create_folder=True):
    config = all_configs[config_id]
    if not create_folder:
        return config
    if trial_id is not None:
        config["trial"] = trial_id
    config = process_config(config)
    revision_hash = get_git_revision_hash()
    if not os.path.exists(os.path.join(config["save_path"], "git_wandb_config.txt")):
        with open(os.path.join(config["save_path"], "git_wandb_config.txt"), "w") as f:
            f.write("To reproduce this experiment, run:\n")
            f.write("\t" + command + "\n")
            f.write("\tgit checkout -b " + config["name"] + " " + revision_hash + "\n")
            f.write("WandB Id: " + config["wandb_id"] + "\n")

    return config


def process_config(config):
    """
    Process the config file to add the trial number and the save path
    :param config: dict
    :return: dict
    """

    config["resumed"] = True
    # If no trial number is specified, create a new one
    if config.get("trial", None) is None:
        config["resumed"] = False
        if os.path.exists(config["save_path"]):
            num_trials = len(os.listdir(config["save_path"]))
            config["trial"] = num_trials + 1
        else:
            config["trial"] = 1

    config["save_path"] = os.path.join(config["save_path"],
                                       "trial_" + str(config["trial"]))
    config["seed"] = config["trial"]
    config["name"] = config["name"] + "/trial_" + str(config["trial"])

    # If the save path does not exist, create it and a new wandb id
    if not os.path.exists(config["save_path"]):
        config["resumed"] = False
        os.makedirs(config["save_path"], exist_ok=True)
        wandb_id = wandb.util.generate_id()
        config["wandb_id"] = wandb_id

    # If a trial number is specified, use the same wandb id
    if config["resumed"]:
        with open(os.path.join(config["save_path"], "git_wandb_config.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                if "WandB Id" in line:
                    config["wandb_id"] = line.split(":")[1].strip()
                    break

    return config


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
