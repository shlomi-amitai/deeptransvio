import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re
from scipy.spatial.transform import Rotation
from matplotlib.widgets import Button

def load_imu_data(imu_file):
    return np.loadtxt(imu_file, usecols=list(range(1, 7)))

def load_poses(pose_file):
    return np.loadtxt(pose_file)

def load_images(image_dir):
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    return image_files

def extract_timestamp(filename):
    match = re.search(r'frame_(\d+)', filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(match.group(1)) * 1e-9  # Convert nanoseconds to seconds

def plot_sequence(imu_data, poses, image_files, start_idx, num_samples):
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle(f"Sequence Visualization (Samples {start_idx} to {start_idx + num_samples - 1})")

    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(imu_data) - start_idx, len(poses) - start_idx, len(image_files) - start_idx)

def plot_sequence(imu_data, poses, image_files, start_idx, num_samples):
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle(f"Sequence Visualization (Samples {start_idx} to {start_idx + num_samples - 1})")

    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(imu_data) - start_idx, len(poses) - start_idx, len(image_files) - start_idx)

    # Plot IMU data
    imu_time = np.arange(num_samples) / 10  # Assuming 10 Hz IMU data
    imu_labels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for i in range(6):
        axes[0].plot(imu_time, imu_data[start_idx:start_idx + num_samples, i], label=imu_labels[i])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('IMU Data')
    axes[0].legend()

    # Extract position and orientation from poses
    positions = poses[start_idx:start_idx + num_samples, [3, 7, 11]]
    rotation_matrices = poses[start_idx:start_idx + num_samples, :12].reshape(-1, 3, 4)[:, :3, :3]
    orientations = np.array([Rotation.from_matrix(R).as_euler('xyz', degrees=True) for R in rotation_matrices])

    # Plot pose data
    pose_time = np.arange(num_samples)  # Assuming poses are at regular intervals
    pos_labels = ['x', 'y', 'z']
    for i in range(3):
        axes[1].plot(pose_time, positions[:, i], label=pos_labels[i])
    axes[1].set_xlabel('Time (steps)')
    axes[1].set_ylabel('Position (m)')
    axes[1].legend()

    # Plot orientation
    ori_labels = ['roll', 'pitch', 'yaw']
    for i in range(3):
        axes[2].plot(pose_time, orientations[:, i], label=ori_labels[i])
    axes[2].set_xlabel('Time (steps)')
    axes[2].set_ylabel('Orientation (degrees)')
    axes[2].legend()

    # Display images
    img = Image.open(image_files[start_idx])
    img_plot = axes[3].imshow(img)
    axes[3].axis('off')
    axes[3].set_title(f"Image at t={extract_timestamp(image_files[start_idx]):.3f}s")

    plt.tight_layout()
    return fig, axes, img_plot

def update_plot(fig, axes, img_plot, imu_data, poses, image_files, start_idx, num_samples):
    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(imu_data) - start_idx, len(poses) - start_idx, len(image_files) - start_idx)

    imu_time = np.arange(num_samples) / 10
    pose_time = np.arange(num_samples)

    # Update IMU plot
    for i in range(6):
        axes[0].lines[i].set_data(imu_time, imu_data[start_idx:start_idx + num_samples, i])

    # Extract position and orientation from poses
    positions = poses[start_idx:start_idx + num_samples, [3, 7, 11]]
    rotation_matrices = poses[start_idx:start_idx + num_samples, :12].reshape(-1, 3, 4)[:, :3, :3]
    orientations = np.array([Rotation.from_matrix(R).as_euler('xyz', degrees=True) for R in rotation_matrices])

    # Update position plot
    for i in range(3):
        axes[1].lines[i].set_data(pose_time, positions[:, i])

    # Update orientation plot
    for i in range(3):
        axes[2].lines[i].set_data(pose_time, orientations[:, i])

    # Update image
    img = Image.open(image_files[start_idx])
    print(image_files[start_idx])
    img_plot.set_data(img)
    axes[3].set_title(f"Image at t={extract_timestamp(image_files[start_idx]):.3f}s")

    # Update x-axis limits for all plots
    for ax in axes[:3]:
        ax.set_xlim(0, num_samples)
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw_idle()
# Main script
data_dir = "/home/ws1/work/Shlomi/deeptransvio/NTNU_rec_data/synced"
sequence = "0"  # Change this to visualize different sequences

imu_file = os.path.join(data_dir, "imus", f"imu_data_{sequence}.txt")
pose_file = os.path.join(data_dir, "poses", "qualisys_ariel_odom_traj_3_id1_synced.kitti")
image_dir = os.path.join(data_dir, "sequences", f"cam0_{sequence}")

imu_data = load_imu_data(imu_file)
poses = load_poses(pose_file)
image_files = load_images(image_dir)

# Initial plot
start_idx = 0
num_samples = 10000
fig, axes, img_plot = plot_sequence(imu_data, poses, image_files, start_idx, num_samples)

class Index:
    def __init__(self, start_idx, num_samples, max_idx):
        self.start_idx = start_idx
        self.num_samples = num_samples
        self.max_idx = max_idx

    def next(self, event):
        self.start_idx = min(self.start_idx + self.num_samples, self.max_idx - self.num_samples)
        update_plot(fig, axes, img_plot, imu_data, poses, image_files, self.start_idx, self.num_samples)

    def prev(self, event):
        self.start_idx = max(0, self.start_idx - self.num_samples)
        update_plot(fig, axes, img_plot, imu_data, poses, image_files, self.start_idx, self.num_samples)

callback = Index(start_idx, num_samples, min(len(imu_data), len(poses), len(image_files)))
ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(ax_next, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(ax_prev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()