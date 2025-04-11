import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re
from scipy.spatial.transform import Rotation
from matplotlib.widgets import Button


def get_data_ranges(imu_data, poses):
    acc_ranges = [
        (np.min(imu_data[:, 0]), np.max(imu_data[:, 0])),
        (np.min(imu_data[:, 1]), np.max(imu_data[:, 1])),
        (np.min(imu_data[:, 2]), np.max(imu_data[:, 2]))
    ]
    gyro_ranges = [
        (np.min(imu_data[:, 3]), np.max(imu_data[:, 3])),
        (np.min(imu_data[:, 4]), np.max(imu_data[:, 4])),
        (np.min(imu_data[:, 5]), np.max(imu_data[:, 5]))
    ]
    pos_ranges = [
        (np.min(poses[:, 3]), np.max(poses[:, 3])),
        (np.min(poses[:, 7]), np.max(poses[:, 7])),
        (np.min(poses[:, 11]), np.max(poses[:, 11]))
    ]
    rotation_matrices = poses[:, :12].reshape(-1, 3, 4)[:, :3, :3]
    orientations = np.array([Rotation.from_matrix(R).as_euler('xyz', degrees=True) for R in rotation_matrices])
    ori_ranges = [
        (np.min(orientations[:, 0]), np.max(orientations[:, 0])),
        (np.min(orientations[:, 1]), np.max(orientations[:, 1])),
        (np.min(orientations[:, 2]), np.max(orientations[:, 2]))
    ]
    return acc_ranges, gyro_ranges, pos_ranges, ori_ranges


def load_imu_data(imu_file):
    imu = sio.loadmat(imu_file)
    acc = imu['imu_data_interp'][:, :3]  # Assuming first 3 columns are accelerometer data
    gyro = imu['imu_data_interp'][:, 3:6]  # Assuming last 3 columns are gyroscope data
    return np.hstack((acc, gyro))

def load_poses(pose_file):
    return np.loadtxt(pose_file)

def load_images(image_dir):
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    return image_files

def extract_timestamp(filename):
    match = re.search(r'(\d{6})', os.path.basename(filename))
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(match.group(1)) * 1e-6  # Convert microseconds to seconds

def plot_sequence(imu_data, poses, image_files, start_idx, num_samples, data_ranges):
    acc_ranges, gyro_ranges, pos_ranges, ori_ranges = data_ranges
    fig = plt.figure(figsize=(15, 30))
    gs = fig.add_gridspec(8, 2)

    axes = []
    # Accelerometer data
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
    # Gyroscope data
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        axes.append(ax)
    # Position data
    for i in range(3):
        ax = fig.add_subplot(gs[3+i, 0])
        axes.append(ax)
    # Orientation data
    for i in range(3):
        ax = fig.add_subplot(gs[3+i, 1])
        axes.append(ax)
    # Start image
    ax_start_img = fig.add_subplot(gs[6:, 0])
    axes.append(ax_start_img)
    # End image
    ax_end_img = fig.add_subplot(gs[6:, 1])
    axes.append(ax_end_img)

    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(poses) - start_idx, len(image_files) - start_idx)
    imu_samples = min(num_samples * 10, len(imu_data) - start_idx * 10)

    # Plot accelerometer data
    imu_time = np.arange(imu_samples) / 100  # Assuming 100 Hz IMU data
    acc_labels = ['acc_x', 'acc_y', 'acc_z']
    acc_colors = ['r', 'g', 'b']
    for i in range(3):
        axes[i].plot(imu_time, imu_data[start_idx*10:start_idx*10 + imu_samples, i], color=acc_colors[i])
        axes[i].set_xlabel('Time (s)', fontsize=8)
        axes[i].set_ylabel(f'{acc_labels[i]} (m/s^2)', fontsize=8)
        axes[i].set_title(f'Accelerometer {acc_labels[i]}', fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].set_ylim(acc_ranges[i])

    # Plot gyroscope data
    gyro_labels = ['gyro_x', 'gyro_y', 'gyro_z']
    gyro_colors = ['c', 'm', 'y']
    for i in range(3):
        axes[i+3].plot(imu_time, imu_data[start_idx*10:start_idx*10 + imu_samples, i+3], color=gyro_colors[i])
        axes[i+3].set_xlabel('Time (s)', fontsize=8)
        axes[i+3].set_ylabel(f'{gyro_labels[i]} (rad/s)', fontsize=8)
        axes[i+3].set_title(f'Gyroscope {gyro_labels[i]}', fontsize=10)
        axes[i+3].tick_params(axis='both', which='major', labelsize=8)
        axes[i+3].set_ylim(gyro_ranges[i])

    # Extract position and orientation from poses
    positions = poses[start_idx:start_idx + num_samples, [3, 7, 11]]
    rotation_matrices = poses[start_idx:start_idx + num_samples, :12].reshape(-1, 3, 4)[:, :3, :3]
    orientations = np.array([Rotation.from_matrix(R).as_euler('xyz', degrees=True) for R in rotation_matrices])

    # Plot pose data
    pose_time = np.arange(num_samples) / 10  # Assuming poses are at 10 Hz
    pos_labels = ['x', 'y', 'z']
    pos_colors = ['r', 'g', 'b']
    for i in range(3):
        axes[i+6].plot(pose_time, positions[:, i], color=pos_colors[i])
        axes[i+6].set_xlabel('Time (s)', fontsize=8)
        axes[i+6].set_ylabel(f'Position {pos_labels[i]} (m)', fontsize=8)
        axes[i+6].set_title(f'Position {pos_labels[i]}', fontsize=10)
        axes[i+6].tick_params(axis='both', which='major', labelsize=8)
        axes[i+6].set_ylim(pos_ranges[i])

    # Plot orientation
    ori_labels = ['roll', 'pitch', 'yaw']
    ori_colors = ['c', 'm', 'y']
    for i in range(3):
        axes[i+9].plot(pose_time, orientations[:, i], color=ori_colors[i])
        axes[i+9].set_xlabel('Time (s)', fontsize=8)
        axes[i+9].set_ylabel(f'Orientation {ori_labels[i]} (degrees)', fontsize=8)
        axes[i+9].set_title(f'Orientation {ori_labels[i]}', fontsize=10)
        axes[i+9].tick_params(axis='both', which='major', labelsize=8)
        axes[i+9].set_ylim(ori_ranges[i])

    # Display start image
    start_img = Image.open(image_files[start_idx])
    start_img_plot = axes[12].imshow(start_img)
    axes[12].axis('off')
    axes[12].set_title(f"Start Image at t={extract_timestamp(image_files[start_idx]):.6f}s", fontsize=10)

    # Display end image
    end_img = Image.open(image_files[start_idx + num_samples - 1])
    end_img_plot = axes[13].imshow(end_img)
    axes[13].axis('off')
    axes[13].set_title(f"End Image at t={extract_timestamp(image_files[start_idx + num_samples - 1]):.6f}s", fontsize=10)

    plt.tight_layout()
    return fig, axes, start_img_plot, end_img_plot

def update_plot(fig, axes, start_img_plot, end_img_plot, imu_data, poses, image_files, start_idx, num_samples, data_ranges):
    acc_ranges, gyro_ranges, pos_ranges, ori_ranges = data_ranges
    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(poses) - start_idx, len(image_files) - start_idx)
    imu_samples = min(num_samples * 10, len(imu_data) - start_idx * 10)

    imu_time = np.arange(imu_samples) / 100
    pose_time = np.arange(num_samples) / 10

    # Update accelerometer plot
    for i in range(3):
        axes[i].lines[0].set_data(imu_time, imu_data[start_idx*10:start_idx*10 + imu_samples, i])
        axes[i].relim()
        axes[i].autoscale_view()

    # Update gyroscope plot
    for i in range(3):
        axes[i+3].lines[0].set_data(imu_time, imu_data[start_idx*10:start_idx*10 + imu_samples, i+3])
        axes[i+3].relim()
        axes[i+3].autoscale_view()

    # Extract position and orientation from poses
    positions = poses[start_idx:start_idx + num_samples, [3, 7, 11]]
    rotation_matrices = poses[start_idx:start_idx + num_samples, :12].reshape(-1, 3, 4)[:, :3, :3]
    orientations = np.array([Rotation.from_matrix(R).as_euler('xyz', degrees=True) for R in rotation_matrices])

    # Update position plot
    for i in range(3):
        axes[i+6].lines[0].set_data(pose_time, positions[:, i])
        axes[i+6].relim()
        axes[i+6].autoscale_view()

    # Update orientation plot
    for i in range(3):
        axes[i+9].lines[0].set_data(pose_time, orientations[:, i])
        axes[i+9].relim()
        axes[i+9].autoscale_view()

    # Update start image
    start_img = Image.open(image_files[start_idx])
    start_img_plot.set_data(start_img)
    axes[12].set_title(f"Start Image at t={extract_timestamp(image_files[start_idx]):.6f}s")

    # Update end image
    end_img = Image.open(image_files[start_idx + num_samples - 1])
    end_img_plot.set_data(end_img)
    axes[13].set_title(f"End Image at t={extract_timestamp(image_files[start_idx + num_samples - 1]):.6f}s")

    # Update x-axis limits for all plots
    for i in range(6):
        axes[i].set_xlim(0, imu_samples / 100)
    for i in range(6, 12):
        axes[i].set_xlim(0, num_samples / 10)

    fig.canvas.draw_idle()
data_dir = "/Users/shlomia/work/my_repo/deeptransvio/data"  # replace with your actual dataset directory path
sequence = 0  # or whatever number you want to use
imu_file = os.path.join(data_dir, "imus", f"{sequence:02d}.mat")
pose_file = os.path.join(data_dir, "poses", f"{sequence:02d}.txt")
image_dir = os.path.join(data_dir, "sequences", f"{sequence:02d}", "image_2")

imu_data = load_imu_data(imu_file)
poses = load_poses(pose_file)
image_files = load_images(image_dir)

# Calculate data ranges
data_ranges = get_data_ranges(imu_data, poses)

# Initial plot
start_idx = 0
num_samples = 10  # You can adjust this value
fig, axes, start_img_plot, end_img_plot = plot_sequence(imu_data, poses, image_files, start_idx, num_samples, data_ranges)


class Index:
    def __init__(self, start_idx, num_samples, max_idx, data_ranges):
        self.start_idx = start_idx
        self.num_samples = num_samples
        self.max_idx = max_idx
        self.data_ranges = data_ranges

    def next(self, event):
        self.start_idx = min(self.start_idx + self.num_samples, self.max_idx - self.num_samples)
        update_plot(fig, axes, start_img_plot, end_img_plot, imu_data, poses, image_files, self.start_idx, self.num_samples, self.data_ranges)

    def prev(self, event):
        self.start_idx = max(0, self.start_idx - self.num_samples)
        update_plot(fig, axes, start_img_plot, end_img_plot, imu_data, poses, image_files, self.start_idx, self.num_samples, self.data_ranges)

callback = Index(start_idx, num_samples, min(len(imu_data), len(poses), len(image_files)), data_ranges)
ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(ax_next, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(ax_prev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()