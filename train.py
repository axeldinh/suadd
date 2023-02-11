import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from configs.experiments import load_config
from configs.wandb import WANDB_PROJECT, ENTITY
from models.lightning_module import LitModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Checkpoint directory")
warnings.filterwarnings("ignore", category=UserWarning, message="The dataloader,")

def main(exp_id, trial_id, epochs, no_wandb, profile, command):

    config = load_config(exp_id, command, trial_id)

    if epochs > 0:
        config["epochs"] = epochs

    if no_wandb:
        config["use_wandb"] = False

    if config["use_wandb"]:
        # Launch wandb before making the model, to load artifacts afterwards
        logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=ENTITY,
            name=config["name"],
            log_model=True,
            save_dir=config["save_path"],
            id=config["wandb_id"],
            resume="must" if config["resumed"] else None,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config["save_path"],
            name="tensorboard",
            log_graph=True,
        )

    model = LitModel(config)

    callbacks = [
        ModelCheckpoint(
            monitor=config["monitor"],
            dirpath=os.path.join(config["save_path"], "checkpoints"),
            filename="best_model",
            save_top_k=1,
            mode=config["monitor_mode"],
        )]

    checkpoint = os.path.join(config["save_path"], "checkpoints", "best_model.ckpt")

    trainer = pl.Trainer(
        accelerator=config["device"],
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config["log_every_n_steps"],
        val_check_interval=config["val_check_interval"],
        auto_lr_find=True,
        auto_scale_batch_size='power',
        num_sanity_val_steps=0,
        profiler="advanced" if profile else None,
    )

    trainer.tune(model)

    model.save_hyperparameters()

    trainer.fit(model, ckpt_path=checkpoint if config["resumed"] else None)

    trainer.test(model, ckpt_path=checkpoint)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", "-e", type=int, default=0, help="Experiment index")
    parser.add_argument("--trial_id", "-t", type=int, help="Trial index")
    parser.add_argument("--no_wandb", "-w", action="store_true", help="Use wandb")
    parser.add_argument("--epochs", "-ep", type=int, help="Number of epochs", default=-1)
    parser.add_argument("--profile", "-p", action="store_true", help="Profile the code")

    args = parser.parse_args()

    # Recover the complete command entered by the user
    command = " ".join(["python"] + [f"--{k} {v}" for k, v in vars(args).items() if v is not None])

    main(**vars(args), command=command)
