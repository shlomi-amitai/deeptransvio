import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os
from datetime import datetime


class image_Inertial_Encoder(nn.Module):
    def __init__(self, opt):
        super(image_Inertial_Encoder, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, opt.i_f_len)  # ResNet18 has 512 features in the last layer
        self.norm = nn.BatchNorm1d(6)
        self.output_size = opt.i_f_len

    def create_inertial_image(self, x):
        """Transform inertial signals to RGB images using Mul2Image encoding"""
        batch_size, seq_len, window_size, channels = x.shape
        # Split accelerometer and gyroscope data
        acc = x[..., :3]  # First 3 channels are accelerometer
        gyro = x[..., 3:]  # Last 3 channels are gyroscope
        # Normalize signals
        acc_norm = torch.nn.functional.normalize(acc, p=2, dim=-1)
        # Create RGB channels through multiplication of acc and gyro
        R = (acc_norm[..., 0] * gyro[..., 0]).unsqueeze(-1)
        G = (acc_norm[..., 1] * gyro[..., 1]).unsqueeze(-1)
        B = (acc_norm[..., 2] * gyro[..., 2]).unsqueeze(-1)
        # Stack RGB channels
        inertial_image = torch.cat([R, G, B], dim=-1)
        # Reshape to [batch_size, seq_len, channels, height]
        inertial_image = inertial_image.permute(0, 1, 3, 2)
        # Scale to [0, 1] range
        inertial_image = (inertial_image - inertial_image.min()) / (inertial_image.max() - inertial_image.min())

        # Visualize the inertial image (uncomment when needed)
        # visualize_inertial_image(inertial_image[0, 0], save_path='debug/inertial_image.png')

        return inertial_image

    def forward(self, x):
        # x: (N, seq_len, 11, 6) - batch_size, sequence_length, window_size, channels
        batch_size, seq_len = x.shape[:2]
        # Create inertial images
        inertial_images = self.create_inertial_image(x)
        # Reshape for ResNet input
        inertial_images = inertial_images.view(-1, 3, inertial_images.shape[-1], 1)
        # Pass through ResNet backbone
        features = self.resnet(inertial_images)
        # Reshape output to match required dimensions
        out = features.view(batch_size, seq_len, self.output_size)
        return out


def visualize_inertial_image(inertial_image, save_path=None):
    # Convert to numpy and move to CPU if on GPU
    img = inertial_image.cpu().numpy()

    # Check the shape of the input
    if len(img.shape) == 2:
        # If it's a 2D image, we can display it directly
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='viridis')  # Using viridis colormap for single-channel image
    elif len(img.shape) == 3:
        # If it's a 3D tensor, assume it's in the format (channels, height, width)
        img = np.transpose(img, (1, 2, 0))  # Reorder to (height, width, channels)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
    else:
        raise ValueError(f"Unexpected shape for inertial_image: {img.shape}")

    plt.title('Inertial Image Visualization')
    plt.axis('off')

    if save_path:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Split the save_path into directory and filename
        save_dir, filename = os.path.split(save_path)

        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)

        # Create the new filename with timestamp
        new_filename = f"{name}_{timestamp}{ext}"

        # Join the directory and new filename
        full_save_path = os.path.join(save_dir, new_filename)

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the figure
        plt.savefig(full_save_path)
        plt.close()
        print(f"Inertial image visualization saved as: {full_save_path}")
    else:
        plt.show()
