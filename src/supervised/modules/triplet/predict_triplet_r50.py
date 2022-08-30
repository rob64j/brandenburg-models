import torch
import argparse
import configparser
import torchmetrics
import numpy as np
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from brandenburg_dataset import BrandenburgDataModule

# from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import (
    SoftmaxEmbedderResNet50,
    TemporalSoftmaxEmbedderResNet50,
)
from pytorch_metric_learning.miners import TripletMarginMiner
from miners import RandomNegativeTripletSelector
from losses import OnlineReciprocalTripletLoss
from src.supervised.utils.model_initialiser import initialise_triplet_model
from sklearn.neighbors import KNeighborsClassifier


class ActionClassifier(pl.LightningModule):
    def __init__(
        self, lr, weight_decay, model_name, freeze_backbone, margin, type_of_triplets, num_classes
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = initialise_triplet_model(
            name=model_name, freeze_backbone=freeze_backbone, out_features=num_classes
        )

        self.classifier = KNeighborsClassifier(n_neighbors=num_classes)

        self.triplet_miner = TripletMarginMiner(
            margin=margin, type_of_triplets=type_of_triplets
        )
        self.triplet_loss = OnlineReciprocalTripletLoss()  # self.selector
        self.ce_loss = nn.CrossEntropyLoss()

        # Training metrics
        self.train_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.train_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average="macro"
        )
        self.train_per_class_acc = torchmetrics.Accuracy(num_classes=num_classes, average="none")

        # Validation metrics
        self.val_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.val_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average="macro"
        )
        self.val_per_class_acc = torchmetrics.Accuracy(num_classes=num_classes, average="none")

    def assign_embedding_name(self, name):
        self.embedding_filename = name

    def forward(self, x):
        emb, pred = self.model(x)
        return emb, pred

    def training_step(self, batch, batch_idx):

        x, y = batch
        embeddings, preds = self(x)

        self.train_top1_acc(preds, y)
        self.train_avg_per_class_acc(preds, y)
        self.train_per_class_acc.update(preds, y)

        a_idx, p_idx, n_idx = self.triplet_miner(embeddings, y)
        labels = torch.cat((y[a_idx], y[p_idx], y[n_idx]), dim=0)

        triplet_loss = self.triplet_loss(
            embeddings[a_idx],
            embeddings[p_idx],
            embeddings[n_idx],
            labels,
        )
        ce_loss = self.ce_loss(preds, y)
        loss = 0.01 * triplet_loss + ce_loss

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        self.log(
            "train_top1_acc",
            self.train_top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log epoch acc
        self.log(
            "train_avg_per_class_acc",
            self.train_avg_per_class_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):

        x, y = batch
        embeddings, preds = self(x)

        self.val_top1_acc(preds, y)
        self.val_avg_per_class_acc(preds, y)
        self.val_per_class_acc.update(preds, y)

        a_idx, p_idx, n_idx = self.triplet_miner(embeddings, y)
        labels = torch.cat((y[a_idx], y[p_idx], y[n_idx]), dim=0)

        triplet_loss = self.triplet_loss(
            embeddings[a_idx],
            embeddings[p_idx],
            embeddings[n_idx],
            labels,
        )
        ce_loss = self.ce_loss(preds, y)
        loss = 0.01 * triplet_loss + ce_loss

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        # Log top-1 acc per epoch
        self.log(
            "val_top1_acc",
            self.val_top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log per class acc per epoch
        self.log(
            "val_avg_per_class_acc",
            self.val_avg_per_class_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def on_predict_epoch_start(self):

        # Embeddings/labels to be stored on the inference set
        self.outputs_embedding = np.zeros((1, 128))
        self.labels_embedding = np.zeros((1))

    def predict_step(self, batch, batch_idx):
        x, y = batch
        embeddings, preds = self(x)
        self.outputs_embedding = np.concatenate(
            (self.outputs_embedding, embeddings.detach().cpu()), axis=0
        )
        self.labels_embedding = np.concatenate(
            (self.labels_embedding, y.detach().cpu()), axis=0
        )

    def on_predict_epoch_end(self, results):
        np.savez(
            self.embedding_filename,
            embeddings=self.outputs_embedding,
            labels=self.labels_embedding,
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_module = BrandenburgDataModule(cfg=cfg)
    model = ActionClassifier.load_from_checkpoint(cfg.get("trainer", "ckpt"))

    name = f"{args.prefix}_{args.split}_embeddings.npz"
    model.assign_embedding_name(name)

    wand_logger = WandbLogger(offline=True)

    if cfg.get("remote", "slurm") == "ssd" or cfg.get("remote", "slurm") == "hdd":
        if not cfg.getboolean("mode", "test"):
            trainer = pl.Trainer(
                gpus=cfg.getint("trainer", "gpus"),
                num_nodes=cfg.getint("trainer", "num_nodes"),
                strategy=cfg.get("trainer", "strategy"),
                max_epochs=cfg.getint("trainer", "max_epochs"),
                stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
                logger=wand_logger,
            )
        else:
            trainer = pl.Trainer(
                gpus=cfg.getint("trainer", "gpus"),
                num_nodes=cfg.getint("trainer", "num_nodes"),
                strategy=cfg.get("trainer", "strategy"),
                max_epochs=cfg.getint("trainer", "max_epochs"),
                stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
                logger=wand_logger,
                fast_dev_run=10,
            )
    else:
        trainer = pl.Trainer(
            gpus=cfg.getint("trainer", "gpus"),
            num_nodes=cfg.getint("trainer", "num_nodes"),
            strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            logger=wand_logger,
            fast_dev_run=5,
        )

    data_module.setup()

    if args.split == "train":
        loader = data_module.train_dataloader()
    elif args.split == "validation":
        loader = data_module.val_dataloader()
    elif args.split == "test":
        loader = data_module.test_dataloader()

    predictions = trainer.predict(model, dataloaders=loader)


if __name__ == "__main__":
    main()
