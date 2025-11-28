import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torchmetrics import MeanMetric
import sys
import os

sys.path.append('../')
from visualize import BatchVisualizer

class LitAutoencoder(pl.LightningModule):
    def __init__(self, architecture, lr=1e-4, step_lr_gamma=0.95):
        super().__init__()
        self.model = architecture
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.step_lr_gamma = step_lr_gamma

        # Metrics
        self.train_mse = MeanMetric()
        self.val_mse = MeanMetric()

        # Placeholder to be replaced by user code
        self.visualizer = BatchVisualizer(m=2, n=8, figsize=(10, 3))

        # Store last validation batch
        self.last_val_batch = None

    def forward(self, x):
        return self.model(x)

    # ------------- TRAINING -----------------
    def training_step(self, batch, batch_idx):
        imgs = batch
        outputs = self(imgs)
        loss = self.loss_fn(outputs, imgs)

        # update metric
        self.train_mse.update(loss)

        # log loss per epoch
        self.log("train/loss_step", loss, on_step=True, prog_bar=False)
        self.log("train/loss_epoch", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # log aggregated metric
        self.log("train/mse_epoch", self.train_mse.compute(), prog_bar=True)
        self.train_mse.reset()

    # ---------------- VALIDATION ------------------
    def validation_step(self, batch, batch_idx):
        imgs = batch
        outputs = self(imgs)
        loss = self.loss_fn(outputs, imgs)

        # update validation metric accumulator
        self.val_mse.update(loss)

        # store last batch for visualization later
        self.last_val_batch = (imgs, outputs)

        self.log("val/loss_step", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_mse_value = self.val_mse.compute()
        self.log("val/mse_epoch", val_mse_value, prog_bar=True)
        self.val_mse.reset()
        
        epoch_num = self.current_epoch
        logger = self.logger
    
        if hasattr(logger, "log_dir"):  
            run_dir = logger.log_dir
        elif hasattr(logger, "save_dir"):
            run_dir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
        else:
            run_dir = "."
        img_dir = os.path.join(run_dir, "val_images")
        os.makedirs(img_dir, exist_ok=True)
    
        if self.last_val_batch is not None:
            imgs, outputs = self.last_val_batch
    
            fig = self.visualizer.visualize_batch(outputs)
            save_path = os.path.join(img_dir, f"pred_epoch_{epoch_num}.png")
            self.visualizer.save_figure(fig[0], save_path)

    # ---------------- OPTIMIZER --------------------
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=1, 
            gamma=self.step_lr_gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mse_epoch",   # Lightning may require this for some schedulers
                "interval": "epoch",
                "frequency": 1,
            },
        }
