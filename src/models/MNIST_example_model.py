from torchvision.models.resnet import ResNet, BasicBlock
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class _MNISTResNet(ResNet):
    """Pytorch ResNet 18, adapted for use with MNIST"""

    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)

        # simply change the first layer to accept greyscale
        self.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )


class MNISTResNetModule(pl.LightningModule):
    """LightningModule implementation of a ResNet for MNIST"""

    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.net = _MNISTResNet()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, targets)

        # return everything so it can be accessed by logging calbacks
        return {
            "loss": loss,
            "logits": logits.detach(),
            "targets": targets,
            "imgs": imgs,
        }

    def training_step_end(self, outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.train_accuracy(outs["logits"], outs["targets"])
        self.log("train/accuracy", self.train_accuracy)
        self.log("train/loss", outs["loss"])

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, targets)

        # return everything so it can be accessed by logging calbacks
        return {
            "loss": loss,
            "logits": logits.detach(),
            "targets": targets,
            "imgs": imgs,
        }

    def validation_step_end(self, outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.valid_accuracy(outs["logits"], outs["targets"])
        self.log("validation/accuracy", self.valid_accuracy)
        self.log("validation/loss", outs["loss"])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNISTResNetModule")
        parser.add_argument(
            "--learning_rate",
            help="learning rate of optimiser",
            type=float,
            default=0.01,
        )
        return parent_parser
