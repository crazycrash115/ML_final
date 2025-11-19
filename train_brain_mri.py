import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from model_brain import BrainTumorConvNet, BrainTumorDeepConvNet


# 1. Paths and transforms
data_dir = "Brain_tumor_images"  

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),        # keeps 3 channels (RGB)
    # !!!!!!KEVIN we can add normalization if you want:
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. Dataset & split (70% train, 15% val, 15% test)
full_dataset = ImageFolder(root=data_dir, transform=transform)
num_classes = len(full_dataset.classes)
print("Classes:", full_dataset.classes, "| num_classes =", num_classes)

train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(0)
)

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
