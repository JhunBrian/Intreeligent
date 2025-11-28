import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class BatchVisualizer:
    def __init__(self, m=2, n=4, figsize=(16, 12)):
        """
        Args:
            m (int): number of rows in the grid
            n (int): number of columns in the grid
            figsize (tuple): figure size
        """
        self.m = m
        self.n = n
        self.figsize = figsize

    def visualize_batch(self, batch):
        """
        Visualize a batch of images.
        Args:
            batch: tensor of images from DataLoader with shape (B, C, H, W)
        Returns:
            fig, axs: matplotlib figure and axes
        """
        # batch is ONLY x_batch now
        x_batch = batch

        if isinstance(x_batch, (list, tuple)) and len(x_batch) == 1:
            # Safety for weird DataLoader wrappers
            x_batch = x_batch[0]

        batch_size = min(len(x_batch), self.m * self.n)

        fig, axs = plt.subplots(self.m, self.n, figsize=self.figsize)
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        for idx in range(batch_size):
            img_tensor = x_batch[idx]

            fig_single = self.visualize(img_tensor)
            fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0)

            img_single = self.fig_to_image(fig_single)
            plt.close(fig_single)

            axs[idx].imshow(img_single)
            axs[idx].axis("off")
            axs[idx].set_aspect('auto')

        # Hide leftover axes
        for j in range(batch_size, len(axs)):
            axs[j].axis("off")

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0.5)
        return fig, axs
        
    @staticmethod
    def visualize(image):
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        ax.axis("off")
        plt.close(fig)
        return fig

    @staticmethod
    def fig_to_image(fig):
        """Convert a Matplotlib figure to a numpy RGB array (updated for Matplotlib >=3.8)."""
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA
        img = buf[:, :, :3]
        return img

    @staticmethod
    def save_figure(fig, fpath):
        image_array = BatchVisualizer.fig_to_image(fig)
        pil_image = Image.fromarray(image_array)
        pil_image.save(fpath)