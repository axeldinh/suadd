import os
import subprocess

from configs.configs_unet import configs as configs_unet

all_configs = {}
all_configs.update(configs_unet)

for exp in all_configs.keys():
    if all_configs[exp].get("trial", None) is None:
        if os.path.exists(all_configs[exp]["save_path"]):
            num_trials = len(os.listdir(all_configs[exp]["save_path"]))
            all_configs[exp]["trial"] = num_trials + 1
        else:
            all_configs[exp]["trial"] = 1
    all_configs[exp]["save_path"] = os.path.join(all_configs[exp]["save_path"],
                                                 "trial_" + str(all_configs[exp]["trial"]))
    os.makedirs(all_configs[exp]["save_path"], exist_ok=True)
    all_configs[exp]["seed"] = all_configs[exp]["trial"]


def load_config(config_id, command):
    config = all_configs[config_id]
    revision_hash = get_git_revision_hash()
    with open(os.path.join(config["save_path"], "git_config.txt"), "w") as f:
        f.write("To reproduce this experiment, run:\n")
        f.write("git checkout -b " + config["name"] + " " + revision_hash + "\n")
        f.write(command + "\n")

    return all_configs[config_id]

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
