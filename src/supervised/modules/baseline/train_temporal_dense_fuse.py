import wandb
import os
import torch
import argparse
import configparser
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import ThreeStreamNetwork


os.environ["WANDB_API_KEY"] = "90100ae7e09e19ac19750449baf59b1441e9a5b8"
os.environ["WANDB_MODE"] = "offline"


class ActionClassifier(pl.LightningModule):
    def __init__(self, lr, weight_decay, freeze_backbone):
        super().__init__()

        wandb.init()

        self.save_hyperparameters()

        self.model = ThreeStreamNetwork(
            device=self.device, freeze_backbone=freeze_backbone
        )

        # Loss
        self.ce_loss = nn.CrossEntropyLoss()

        # Training metrics
        self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.train_per_class_accuracy = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )
        # Validation metrics
        self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)
        self.val_per_class_accuracy = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, y = batch
        pred = self(x)

        self.top1_train_accuracy(pred, y)
        self.train_per_class_accuracy(pred, y)

        loss = self.ce_loss(pred, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        self.log(
            "train_top1_acc_epoch",
            self.top1_train_accuracy,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            "train_per_class_acc_epoch",
            self.train_per_class_accuracy,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss_epoch",
            loss,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        self.top1_val_accuracy(pred, y)
        self.val_per_class_accuracy(pred, y)

        loss = self.ce_loss(pred, y)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        self.log(
            "val_top1_acc_epoch",
            self.top1_val_accuracy,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        # Log per class acc per epoch
        self.log(
            "val_per_class_acc",
            self.val_per_class_accuracy,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_module = SupervisedPanAfDataModule(cfg=cfg)

    model = ActionClassifier(
        lr=cfg.getfloat("hparams", "lr"),
        weight_decay=cfg.getfloat("hparams", "weight_decay"),
        freeze_backbone=cfg.getboolean("hparams", "freeze_backbone"),
    )

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_top1_acc", monitor="val_top1_acc_epoch", mode="max"
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_per_class_acc",
        monitor="val_per_class_acc",
        mode="max",
    )

    if cfg.get("remote", "slurm") == "ssd" or cfg.get("remote", "slurm") == "hdd":
        if not cfg.getboolean("mode", "test"):
            trainer = pl.Trainer(
                gpus=cfg.getint("trainer", "gpus"),
                num_nodes=cfg.getint("trainer", "num_nodes"),
                strategy=cfg.get("trainer", "strategy"),
                max_epochs=cfg.getint("trainer", "max_epochs"),
                stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
                callbacks=[
                    val_top1_acc_checkpoint_callback,
                    val_per_class_acc_checkpoint_callback,
                ],
            )
        else:
            trainer = pl.Trainer(
                gpus=cfg.getint("trainer", "gpus"),
                num_nodes=cfg.getint("trainer", "num_nodes"),
                strategy=cfg.get("trainer", "strategy"),
                max_epochs=cfg.getint("trainer", "max_epochs"),
                stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
                fast_dev_run=10,
            )
    else:
        trainer = pl.Trainer(
            gpus=cfg.getint("trainer", "gpus"),
            num_nodes=cfg.getint("trainer", "num_nodes"),
            strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            fast_dev_run=5,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
