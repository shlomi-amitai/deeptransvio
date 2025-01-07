import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os
from datetime import datetime


class image_Inertial_Encoder(nn.Module):
    def __init__(self, opt):
        super(image_Inertial_Encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(opt.seq_len, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        
        # Calculate the flattened feature size
        self.flattened_size = 256 * 11 * 6
        
        # Modify the projection layer to output the correct size
        self.proj = nn.Linear(self.flattened_size, opt.seq_len * opt.i_f_len)
        
        self.output_size = opt.i_f_len
        self.seq_len = opt.seq_len

    def create_inertial_image(self, x):
        """Transform inertial signals to images using raw IMU data"""
        batch_size, seq_len, window_size, channels = x.shape

        # Reshape to use seq_len as channels: [batch_size, seq_len, 11, 6]
        inertial_image = x.permute(0, 1, 2, 3)

        # Scale to [0, 1] range
        inertial_image = (inertial_image - inertial_image.min()) / (inertial_image.max() - inertial_image.min())

        return inertial_image

    def forward(self, x):
        device = x.device
        self.to(device)
        
        batch_size, seq_len = x.shape[:2]
        
        inertial_images = self.create_inertial_image(x)
        
        # Add this line to visualize and save the inertial images
        visualize_inertial_image(inertial_images, save_path='debug_images/inertial_images.png')
        
        inertial_images = inertial_images.permute(0, 1, 2, 3)
        
        self.encoder_conv[0] = nn.Conv2d(seq_len, 64, kernel_size=3, padding=1).to(device)
        
        features = self.encoder_conv(inertial_images)
        
        # Flatten the features
        features = features.view(batch_size, -1)
        
        # Project to desired output size
        out = self.proj(features)
        
        # Reshape to match the original batch size and sequence length
        out = out.view(batch_size, self.seq_len, self.output_size)
        return out



import matplotlib.pyplot as plt
import os
from datetime import datetime

import math

def visualize_inertial_image(inertial_images, save_path=None):
    # Convert to numpy and move to CPU if on GPU
    imgs = inertial_images.cpu().numpy()

    # Check the shape of the input
    if len(imgs.shape) == 4:  # (batch_size, seq_len, 11, 6)
        batch_size, seq_len, height, width = imgs.shape
    else:
        raise ValueError(f"Unexpected shape for inertial_images: {imgs.shape}")

    for batch_idx in range(batch_size):
        # Calculate the number of rows and columns for the subplot grid
        n_cols = min(3, seq_len)
        n_rows = math.ceil(seq_len / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        fig.suptitle(f'Inertial Image Visualization - Batch {batch_idx}')

        # Ensure axes is always a 2D array
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for seq_idx in range(seq_len):
            row = seq_idx // n_cols
            col = seq_idx % n_cols
            ax = axes[row, col]
            
            img = imgs[batch_idx, seq_idx]
            im = ax.imshow(img, cmap='viridis', aspect='auto')
            ax.set_title(f'Sequence {seq_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Hide any unused subplots
        for idx in range(seq_len, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Split the save_path into directory and filename
            save_dir, filename = os.path.split(save_path)

            # Split the filename into name and extension
            name, ext = os.path.splitext(filename)

            # Create the new filename with timestamp and batch index
            new_filename = f"{name}_batch{batch_idx}_{timestamp}{ext}"

            # Join the directory and new filename
            full_save_path = os.path.join(save_dir, new_filename)

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Save the figure
            plt.savefig(full_save_path)
            plt.close()
            print(f"Inertial image visualization for batch {batch_idx} saved as: {full_save_path}")
        else:
            plt.show()

