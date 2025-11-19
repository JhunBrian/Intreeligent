import torch
import torch.nn.functional as F

class MaskResize:
    def __init__(self, size=(500, 500)):
        self.size = size  # (height, width)

    def __call__(self, target):
        h, w = target['masks'].shape[-2:]  # original size
        new_h, new_w = self.size

        # --- Resize masks ---
        masks = target['masks'].unsqueeze(1).float()  # add channel dim for interpolate
        masks = F.interpolate(masks, size=(new_h, new_w), mode='nearest')  
        masks = masks.squeeze(1).byte()  # back to [N, H, W]

        # --- Scale boxes ---
        boxes = target['boxes'].clone()
        scale_x = new_w / w
        scale_y = new_h / h
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        # --- Recompute area (from masks) ---
        area = masks.sum(dim=(1, 2)).float()

        # --- Return resized target ---
        return {
            'boxes': boxes,
            'labels': target['labels'],
            'masks': masks,
            'image_id': target['image_id'],
            'area': area,
            'iscrowd': target['iscrowd']
        }