from lightning.pytorch import LightningModule
from torch import nn, Tensor
import torch
import torchmetrics
from typing import Tuple

class BaseModel(LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.model = self.build_model()

    def build_model(self) -> nn.Module:
        raise Exception("Not yet implemented")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        return nn.functional.cross_entropy(logits, target)

    def shared_step(self, mode: str, batch: Tuple[Tensor, Tensor], batch_index: int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.accuracy(output, target)

        self.log(f"{mode}_step_acc", self.accuracy, prog_bar=True)
        self.log(f"{mode}_step_loss", loss, prog_bar=False)
        return loss

    def training_step(self, batch, batch_index):
        return self.shared_step('train', batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.shared_step('val', batch, batch_index)

    def test_step(self, batch, batch_index):
        return self.shared_step('test', batch, batch_index)


class BrainTumorConvNet(BaseModel):
    """
    Baseline CNN for 128x128 RGB brain MRI.
    Input: (B, 3, 128, 128)
    """
    def build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 16 x 64 x 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32 x 32 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64 x 16 x 16

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )


class BrainTumorDeepConvNet(BaseModel):
    """
    Deeper CNN with more channels + dropout.
    Use this as your 'better' model to compare against the baseline.
    """
    def build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 128 x 16 x 16

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
