import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from model_brain import BrainTumorConvNet, BrainTumorDeepConvNet

class CleanImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        if ".ipynb_checkpoints" in classes:
            classes.remove(".ipynb_checkpoints")
            class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

# 1. Paths and transforms
data_dir = "data/Brain_MRI_Images/Train"  
val_dir = "data/Brain_MRI_Images/Validation"

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),        # keeps 3 channels (RGB)
    # !!!!!!KEVIN we can add normalization if you want:
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. Load train and validation datasets separately
train_dataset = CleanImageFolder(root=data_dir, transform=transform)
val_dataset = CleanImageFolder(root=val_dir, transform=transform)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes, "| num_classes =", num_classes)

# Split training data for test set (85% train, 15% test)
train_size = int(0.85 * len(train_dataset))
test_size = len(train_dataset) - train_size

train_dataset, test_dataset = random_split(
    train_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(0)
)
# `random_split` is a function from PyTorch that is used to split a
# dataset into random train and test subsets. It takes the dataset to be
# split, along with a list of sizes for each subset. The function then
# returns two new datasets, one for training and one for testing, with
# the data randomly distributed according to the specified sizes.

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers=2)
test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def train_model(model, train_loader, val_loader, model_name: str, max_epochs: int = 10):
    # clean old logs
    shutil.rmtree(f"./lightning_logs/{model_name}", ignore_errors=True)

    seed_everything(0, workers=True)
    logger = CSVLogger("./lightning_logs", name=model_name)

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Save final weights
    torch.save(model.state_dict(), f"{model_name}.pth")

    # Evaluate on test set
    test_results = trainer.test(model, dataloaders=test_dataloader)
    print(f"Test results for {model_name}:", test_results)

    return model_name


def show_metrics(name: str):

    metrics_path = f"./lightning_logs/{name}/version_0/metrics.csv"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    df = pd.read_csv(metrics_path)
    df.set_index("step", inplace=True)

    ax = df[['train_step_acc']].dropna().plot()
    df[['val_step_acc']].dropna().plot(ax=ax)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(name)
    plt.legend()
    plt.show()

    return df[['val_step_acc']].dropna().round(3)


# 4. Train baseline model
baseline_model = BrainTumorConvNet(num_classes=num_classes).to(device)
baseline_name = "BrainTumorConvNet"

baseline_model_name = train_model(
    model=baseline_model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    model_name=baseline_name,
    max_epochs=10
)

baseline_val = show_metrics(baseline_model_name)
print("Baseline val acc (last few steps):")
print(baseline_val.tail())


# 5. Train deeper model for comparison
deep_model = BrainTumorDeepConvNet(num_classes=num_classes).to(device)
deep_name = "BrainTumorDeepConvNet"

deep_model_name = train_model(
    model=deep_model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    model_name=deep_name,
    max_epochs=10
)

deep_val = show_metrics(deep_model_name)
print("Deep model val acc (last few steps):")
print(deep_val.tail())
