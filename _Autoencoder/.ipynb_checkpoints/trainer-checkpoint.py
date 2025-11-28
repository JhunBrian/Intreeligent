import torch
import torch.nn as nn
import pytorch_lightning as pl


class LitAutoencoder(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module = None,
                 optimizer_class=torch.optim.Adam,
                 optimizer_params: dict = None,
                 lr: float = 1e-4,
                 visualize: bool = True):
        """
        Flexible Lightning module for any autoencoder model.

        Args:
            model (nn.Module): PyTorch model (e.g., UNetAutoencoder, ResNet34Autoencoder)
            loss_fn (nn.Module, optional): Loss function (default: MSELoss)
            optimizer_class (torch.optim.Optimizer): Optimizer class (default: Adam)
            optimizer_params (dict, optional): Extra optimizer params (e.g., weight_decay)
            lr (float): Learning rate
            visualize (bool): Whether to use BatchViz to visualize validation reconstructions
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.visualize = visualize
        if self.visualize:
            self.batch_viz = BatchViz()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.loss_fn(outputs, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.loss_fn(outputs, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        # visualize first batch
        if self.visualize and batch_idx == 0:
            self.batch_viz.visualize_batch(outputs)

        return loss

    def configure_optimizers(self):
        optimizer_kwargs = {"params": self.parameters(), "lr": self.hparams.lr, **self.optimizer_params}
        return self.optimizer_class(**optimizer_kwargs)
