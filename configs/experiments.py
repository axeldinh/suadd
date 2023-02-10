from configs.configs_unet import configs as configs_unet

all_configs = {}
all_configs.update(configs_unet)


def load_config(config_id):
    return all_configs[config_id]
