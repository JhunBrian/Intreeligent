import pandas as pd
from torch.utils.data import Dataset
import cv2

class AEDataset(Dataset):
    def __init__(self, image_data: pd.DataFrame, threshold=None, transforms=None):
        if threshold is not None:
            df = image_data.query("height > @threshold and width > @threshold")
        else:
            df = image_data

        self.image_data = df[['version', 'image_path', 'height']].sort_values(
            by='height'
        ).reset_index(drop=True)

        self.image_paths = self.image_data['image_path'].tolist()

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def __len__(self):
        return len(self.image_paths)