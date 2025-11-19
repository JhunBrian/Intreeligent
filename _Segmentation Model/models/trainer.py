import os
from PIL import Image
import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from postprocess import PredictionProcessor

class InstanceSegLightningModule(pl.LightningModule):
    def __init__(self, model, batch_visualizer, mask_threshold=0.5, score_threshold=0.5):
        super().__init__()
        self.model = model
        self.map_metric = MeanAveragePrecision(iou_type="segm")
        self.pred_processor = PredictionProcessor(mask_threshold, score_threshold)
        self.last_val_batch = None
        self.batch_visualizer = batch_visualizer
        self.logged_gt = False

    def forward(self, images):
        # self.model.eval()
        # with torch.no_grad():
        raw_pred = self.model(images)  # dict if batch_size=1, else list of dicts
        if isinstance(raw_pred, dict):
            raw_pred = [raw_pred]  # wrap single image in list
        processed_preds = [self.pred_processor(p) for p in raw_pred]
        return processed_preds

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
    
        if isinstance(targets, dict):
            targets = [targets]
    
        loss_dict = self.model(images, targets)
        if isinstance(loss_dict, dict):
            val_loss = sum(loss_dict.values())
        else:
            val_loss = torch.tensor(0.0, device=images[0].device)
    
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
    
        preds = self.forward(images)
        self.map_metric.update(preds, targets)
    
        # store the last batch for visualization
        self.last_val_batch = (images, targets, preds)
    
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.map_metric.reset()

        if self.last_val_batch is None:
            return

        x, y, processed_preds = self.last_val_batch
        bv = self.batch_visualizer
        epoch_num = self.current_epoch  

        if self.logger is not None:
            save_dir = os.path.join(self.logger.log_dir, "images")
        else:
            save_dir = "./"  # fallback

        os.makedirs(save_dir, exist_ok=True)

        if not self.logged_gt:
            f_gt = bv.visualize_batch((x, y))
            gt_path = os.path.join(save_dir, "ground_truth_once.png")
            Image.fromarray(bv.fig_to_image(f_gt[0])).save(gt_path)

            self.logged_gt = True  # <-- prevents duplicate saves

        f_pred = bv.visualize_batch((x, processed_preds))
        pred_path = os.path.join(save_dir, f"pred_epoch_{epoch_num}.png")
        Image.fromarray(bv.fig_to_image(f_pred[0])).save(pred_path)

        self.last_val_batch = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }