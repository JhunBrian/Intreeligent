import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils

class TreeDataset(Dataset):
    def __init__(self, root, subset, image_trans=None, mask_trans=None):
        """
        Args:
            root (str): dataset root path
            subset (str): 'train', 'val', or 'test'
        """
        self.root_dir = os.path.join(root, subset)
        self.ann_file = os.path.join(self.root_dir, "_annotations.coco.json")

        self.image_trans = image_trans
        self.mask_trans = mask_trans

        with open(self.ann_file, "r") as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.annotations = {}
        for ann in self.coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.ids = [img_id for img_id in self.images.keys() if img_id in self.annotations]

    def __len__(self):
        return len(self.ids)

    # def __getitem__(self, idx):
    #     img_id = self.ids[idx]
    #     img_info = self.images[img_id]
    #     annots = self.annotations.get(img_id, [])
    
    #     img_path = os.path.join(self.root_dir, img_info["file_name"])
    #     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    #     boxes, labels, masks, areas, iscrowd = [], [], [], [], []
    
    #     for ann in annots:
    #         x, y, w, h = ann["bbox"]
    #         boxes.append([x, y, x + w, y + h])
    #         labels.append(ann["category_id"])
    #         areas.append(ann["area"])
    #         iscrowd.append(ann.get("iscrowd", 0))
    
    #         seg = ann["segmentation"]
    #         mask = self._polygons_to_mask(seg, img_info["height"], img_info["width"])
    #         masks.append(mask)
    
    #     # Convert masks list to a single numpy array (efficient)
    #     if len(masks) > 0:
    #         masks_np = np.stack(masks, axis=0)  # shape: (num_objects, H, W)
    #         masks_tensor = torch.as_tensor(masks_np, dtype=torch.uint8)
    #     else:
    #         masks_tensor = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
    
    #     target = {
    #         "boxes": torch.as_tensor(boxes, dtype=torch.float32),
    #         "labels": torch.as_tensor(labels, dtype=torch.int64),
    #         "masks": masks_tensor,
    #         "image_id": torch.tensor([img_id]),
    #         "area": torch.as_tensor(areas, dtype=torch.float32),
    #         "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
    #     }
    
    #     if self.image_trans:
    #         img = self.image_trans(img)
    #     if self.mask_trans:
    #         target = self.mask_trans(target)
    
    #     return img, target

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # If it's a slice, return a list of items
            return [self[i] for i in range(*idx.indices(len(self)))]
        
        # Original single index behavior
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        annots = self.annotations.get(img_id, [])
        
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        
        for ann in annots:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
        
            seg = ann["segmentation"]
            mask = self._polygons_to_mask(seg, img_info["height"], img_info["width"])
            masks.append(mask)
        
        if len(masks) > 0:
            masks_np = np.stack(masks, axis=0)
            masks_tensor = torch.as_tensor(masks_np, dtype=torch.uint8)
        else:
            masks_tensor = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        
        if self.image_trans:
            img = self.image_trans(img)
        if self.mask_trans:
            target = self.mask_trans(target)
        
        return img, target


    def _polygons_to_mask(self, polygons, height, width):
        """Convert segmentation polygons to binary mask."""
        rles = maskUtils.frPyObjects(polygons, height, width)
        rle = maskUtils.merge(rles)
        return maskUtils.decode(rle).astype(np.uint8)