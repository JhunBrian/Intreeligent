import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import random

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
        Visualize a batch using the class instance method.
        Args:
            batch: tuple (x, y) from DataLoader
        Returns:
            fig, axs: matplotlib figure and axes
        """
        x_batch, y_batch = batch
        batch_size = min(len(x_batch), self.m * self.n)
    
        fig, axs = plt.subplots(self.m, self.n, figsize=self.figsize)
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
        for idx in range(batch_size):
            sample = (x_batch[idx], y_batch[idx])
            fig_single = self.visualize(sample)           
            
            fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            img_single = self.fig_to_image(fig_single)
            plt.close(fig_single)
            
            axs[idx].imshow(img_single)
            axs[idx].axis("off")
            axs[idx].set_aspect('auto')
    
        for j in range(batch_size, len(axs)):
            axs[j].axis("off")
    
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0.5)
        return fig, axs

    @staticmethod
    def visualize(sample):
        """
        Visualize an instance (image + masks + boxes) and return a matplotlib figure.
        
        Args:
            sample (tuple): (image_tensor, target_dict)
        """
        image, target = sample
    
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # CxHxW -> HxWxC
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
    
        for i in range(len(target["boxes"])):
            mask = target["masks"][i].cpu().numpy().squeeze()  # [H, W]
            color = np.array([random.random(), random.random(), random.random(), 0.5])  # RGBA
            ax.imshow(np.dstack((mask * color[0],
                                 mask * color[1],
                                 mask * color[2],
                                 mask * color[3])))
    
            x1, y1, x2, y2 = target["boxes"][i].cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        ax.axis("off")
        plt.close(fig)
        return fig

    @staticmethod
    def fig_to_image(fig):
        """Convert a Matplotlib figure to a numpy RGB array (updated for Matplotlib >=3.8)."""
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA
        img = buf[:, :, :3]  # Drop alpha
        return img