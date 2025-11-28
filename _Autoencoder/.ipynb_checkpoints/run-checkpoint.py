import os
import cv2
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_lightning.loggers import CSVLogger
from tqdm.notebook import tnrange, tqdm_notebook

from models.resnets import ResNetAutoencoder
from models.trainer import LitAutoencoder
from dataset import AEDataset

sys.path.append("../_Segmentation Model/")
sys.path.append("../")

import utils
from transforms import MaskResize, InstanceMaskCropper, PadToSizeTensor, MinMaxNormalize

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)


def build_trainer(
    max_epochs=30,
    monitor_metric="val/mse_epoch",
    save_top_k=3,
    early_stop_patience=5,
    default_root_dir="runs/",
    gradient_clip_val=1.0,
):

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename="autoencoder-{epoch:02d}-{val_mse:.4f}",
        save_top_k=save_top_k,
        mode="min",
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor=monitor_metric,
        patience=early_stop_patience,
        mode="min",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    csv_logger = CSVLogger(
        save_dir=default_root_dir,
        name="autoencoder_logs"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        gradient_clip_val=gradient_clip_val,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    return trainer



# -------------------------
# SAFE EXECUTION BLOCK
# -------------------------
if __name__ == "__main__":

    image_data = pd.read_csv('image_data.csv')

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(512, 512)),
        MinMaxNormalize()
    ])

    utils.set_seed()
    batch_size = 16

    all_dataset = AEDataset(image_data, 250, trans)
    all_dataset.image_data = all_dataset.image_data[
        all_dataset.image_data['version'] == '_separated'
    ]

    train_set, val_set, test_set = random_split(all_dataset, [0.7, 0.2, 0.1])

    print(f"Train Set: {len(train_set)}")
    print(f"Validation Set: {len(val_set)}")
    print(f"Test Set: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    arch = ResNetAutoencoder(model_name='resnet34', pretrained=False)
    model = LitAutoencoder(arch, lr=1e-4)

    trainer = build_trainer(
        max_epochs=30,
        monitor_metric="val/mse_epoch",
        save_top_k=3,
        early_stop_patience=8,
        default_root_dir="results/"
    )

    trainer.fit(model, train_loader, val_loader)