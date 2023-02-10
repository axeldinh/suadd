from argparse import ArgumentParser
import pytorch_lightning as pl

from configs.experiments import load_config
from lightning_module import LitModel


def main(exp_idx):
    config = load_config(exp_idx)

    model = LitModel(config)

    trainer = pl.Trainer(
        accelerator=config["device"],
        max_epochs=config["epochs"],
    )

    trainer.fit(model)

    trainer.test(model)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--exp_idx", "-e", type=int, default=0, help="Experiment index")

    args = parser.parse_args()

    main(**vars(args))
