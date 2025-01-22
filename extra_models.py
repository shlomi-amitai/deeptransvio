import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def high_pass_filter(data, alpha=0.8):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0] - np.mean(data)
    for i in range(1, len(data)):
        filtered_data[i] = alpha * (filtered_data[i-1] + data[i] - data[i-1])
    return filtered_data

def preprocess_inertial_data(inertial_images):
    # Assuming inertial_images shape is (batch_size, seq_len, 11, 6)
    batch_size, seq_len, height, channels = inertial_images.shape
    
    # Create a new array to store the processed data
    processed_data = np.zeros_like(inertial_images.cpu().numpy())
    
    for b in range(batch_size):
        for h in range(seq_len):
            processed_data[b, h, :, :] = inertial_images[b, h, :, :].cpu().numpy()
            # Apply high-pass filter to acceleration data (first 3 channels)
            processed_data[b, h, :, 2] = high_pass_filter(inertial_images[b, h, :, 2].cpu().numpy())

            # Copy gyroscope data as is (last 3 channels)
            # processed_data[b, h, :, 3:] = inertial_images[b, h, :, 3:].cpu().numpy()
    
    return torch.from_numpy(processed_data).to(inertial_images.device)
def create_inertial_image(self, x):
    """Transform inertial signals to images using preprocessed IMU data"""
    batch_size, seq_len, window_size, channels = x.shape

    # Preprocess the inertial data
    x = preprocess_inertial_data(x)

    # Reshape to use seq_len as channels: [batch_size, seq_len, 11, 6]
    inertial_image = x.permute(0, 1, 2, 3)

    # Scale to [0, 1] range
    inertial_image = (inertial_image - inertial_image.min()) / (inertial_image.max() - inertial_image.min())

    return inertial_image
class image_Inertial_Encoder(nn.Module):
    def __init__(self, opt):
        super(image_Inertial_Encoder, self).__init__()

        # Separate encoders for accelerometer and gyroscope data
        self.accel_encoder = self.create_encoder(10, opt.imu_dropout)
        self.gyro_encoder = self.create_encoder(10, opt.imu_dropout)

        # Calculate the flattened feature size for each encoder
        self.flattened_size = 256 * 11 * 3  # 3 instead of 6 as we're processing 3 channels each

        # Combine features from both encoders
        self.combiner = nn.Sequential(
            nn.Linear(2 * self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(opt.imu_dropout)
        )

        # Final projection layer
        self.proj = nn.Linear(512, opt.seq_len * opt.i_f_len)
        
        self.output_size = opt.i_f_len
        self.seq_len = opt.seq_len

    def create_encoder(self, input_channels, dropout):
        return nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

    def create_inertial_image(self, x):
        """Transform inertial signals to images using preprocessed IMU data"""
        batch_size, seq_len, window_size, channels = x.shape

        # Preprocess the inertial data
        x = preprocess_inertial_data(x)

        # Separate accelerometer and gyroscope data
        accel_data = x[:, :, :, :3]
        gyro_data = x[:, :, :, 3:]

        # Create accelerometer image
        accel_image = accel_data.permute(0, 1, 2, 3)
        accel_image = (accel_image - accel_image.min()) / (accel_image.max() - accel_image.min())

        # Create gyroscope image
        gyro_image = gyro_data.permute(0, 1, 2, 3)
        gyro_image = (gyro_image - gyro_image.min()) / (gyro_image.max() - gyro_image.min())

        return accel_image, gyro_image

    def forward(self, x):
        device = x.device
        self.to(device)
        
        batch_size, seq_len = x.shape[:2]

        accel_image, gyro_image = self.create_inertial_image(x)

        # visualize_inertial_image(accel_image, save_path='debug_images/accel_image.png')
        # visualize_inertial_image(gyro_image, save_path='debug_images/gyro_image.png')

        # Process accelerometer and gyroscope data separately
        accel_features = self.accel_encoder(accel_image)
        gyro_features = self.gyro_encoder(gyro_image)
        
        # Flatten and combine features
        accel_features = accel_features.view(batch_size, -1)
        gyro_features = gyro_features.view(batch_size, -1)
        combined_features = torch.cat([accel_features, gyro_features], dim=1)
        
        # Combine features
        combined_features = self.combiner(combined_features)
        
        # Project to desired output size
        out = self.proj(combined_features)
        
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

