import torch
import json
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer


class PerClassAccuracy(Callback):
    def __init__(
        self,
        which_classes,
        root_dir,
    ):
        super().__init__()

        # self.majority_classes = ["sitting", "standing", "walking"]
        # self.classes = [
        #     "camera_interaction",
        #     "climbing_down",
        #     "climbing_up",
        #     "hanging",
        #     "running",
        #     "sitting",
        #     "sitting_on_back",
        #     "standing",
        #     "walking",
        # ]

        # self.initialise_classes(which_classes=which_classes)
        self.classes = self.extract_from_json(root_dir=root_dir)


    def extract_from_json(self, root_dir):
        try:
            with open (root_dir + '/species_dict.json', 'rb') as f:
                s_dict = json.load(f)
        except:
            print("No species dict found in root dir: " + root_dir)
            exit()
        return dict(enumerate(list(s_dict.keys())))

    # def initialise_classes(self, which_classes):
    #     if which_classes == "majority":
    #         self.classes = self.majority_classes
    #     elif which_classes == "minority":
    #         self.classes = [x for x in self.classes if x not in self.majority_classes]
    #     self.classes = dict(enumerate(self.classes))

    def per_class_dict(self, x: torch.Tensor):

        results = {}
        x = torch.nan_to_num(x)
        for i, item in enumerate(x):
            results[self.classes[i]] = float(item)
        return results

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        train_per_class_acc = self.per_class_dict(
            pl_module.train_per_class_acc.compute()
        )
        pl_module.log("train_per_class_acc", train_per_class_acc, on_epoch=True)
        pl_module.train_per_class_acc.reset()

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        val_per_class_acc = self.per_class_dict(pl_module.val_per_class_acc.compute())
        pl_module.log("val_per_class_acc", val_per_class_acc, on_epoch=True)
        pl_module.val_per_class_acc.reset()
