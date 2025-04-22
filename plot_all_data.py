import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re
from scipy.spatial.transform import Rotation
from matplotlib.widgets import Button


def get_data_ranges(imu_data, poses, ahrs_data):
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
    ahrs_ranges = [
        (np.min(ahrs_data[:, 0]), np.max(ahrs_data[:, 0])),
        (np.min(ahrs_data[:, 1]), np.max(ahrs_data[:, 1])),
        (np.min(ahrs_data[:, 2]), np.max(ahrs_data[:, 2]))
    ]
    return acc_ranges, gyro_ranges, pos_ranges, ori_ranges, ahrs_ranges


def load_imu_data(imu_file):
    imu = sio.loadmat(imu_file)
    acc = imu['imu_data_interp'][:, :3]  # Assuming first 3 columns are accelerometer data
    gyro = imu['imu_data_interp'][:, 3:6]  # Assuming last 3 columns are gyroscope data
    return np.hstack((acc, gyro))


def load_poses(pose_file):
    return np.loadtxt(pose_file)


import numpy as np
from scipy.spatial.transform import Rotation


def generate_ahrs_data(poses):
    rotation_matrices = poses[:, :9].reshape(-1, 3, 3)
    euler_angles = Rotation.from_matrix(rotation_matrices).as_euler('xyz', degrees=True)

    # Ensure yaw is in the range [-180, 180]
    euler_angles[:, 2] = (euler_angles[:, 2] + 180) % 360 - 180

    return euler_angles


def load_images(image_dir):
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    return image_files


def extract_timestamp(filename):
    match = re.search(r'(\d{6})', os.path.basename(filename))
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(match.group(1)) * 1e-6  # Convert microseconds to seconds


def plot_sequence(imu_data, poses, image_files, ahrs_data, start_idx, num_samples, data_ranges):
    acc_ranges, gyro_ranges, pos_ranges, ori_ranges, ahrs_ranges = data_ranges
    fig = plt.figure(figsize=(20, 35))  # Adjusted figure height
    gs = fig.add_gridspec(8, 3, height_ratios=[1, 1, 1, 1, 1, 2, 0.1, 1])  # 8 rows, 3 columns

    axes = []
    # Accelerometer data
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    # Gyroscope data
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        axes.append(ax)
    # Position data
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        axes.append(ax)
    # Orientation data
    for i in range(3):
        ax = fig.add_subplot(gs[3, i])
        axes.append(ax)
    # AHRS data
    for i in range(3):
        ax = fig.add_subplot(gs[4, i])
        axes.append(ax)

    # Start image
    ax_start_img = fig.add_subplot(gs[5, 0:3:2])  # Span two columns
    axes.append(ax_start_img)
    # End image
    ax_end_img = fig.add_subplot(gs[5, 1:3])  # Span two columns
    axes.append(ax_end_img)

    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(poses) - start_idx, len(image_files) - start_idx)
    imu_samples = min(num_samples * 10, len(imu_data) - start_idx * 10)

    # Plot accelerometer data
    imu_time = np.arange(imu_samples) / 100  # Assuming 100 Hz IMU data
    acc_labels = ['acc_x', 'acc_y', 'acc_z']
    acc_colors = ['r', 'g', 'b']
    for i in range(3):
        axes[i].plot(imu_time, imu_data[start_idx * 10:start_idx * 10 + imu_samples, i], color=acc_colors[i])
        axes[i].set_xlabel('Time (s)', fontsize=8)
        axes[i].set_ylabel(f'{acc_labels[i]} (m/s^2)', fontsize=8)
        axes[i].set_title(f'Accelerometer {acc_labels[i]}', fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].set_ylim(acc_ranges[i])

    # Plot gyroscope data
    gyro_labels = ['gyro_x', 'gyro_y', 'gyro_z']
    gyro_colors = ['c', 'm', 'y']
    for i in range(3):
        axes[i + 3].plot(imu_time, imu_data[start_idx * 10:start_idx * 10 + imu_samples, i + 3], color=gyro_colors[i])
        axes[i + 3].set_xlabel('Time (s)', fontsize=8)
        axes[i + 3].set_ylabel(f'{gyro_labels[i]} (rad/s)', fontsize=8)
        axes[i + 3].set_title(f'Gyroscope {gyro_labels[i]}', fontsize=10)
        axes[i + 3].tick_params(axis='both', which='major', labelsize=8)
        axes[i + 3].set_ylim(gyro_ranges[i])

    # Extract position and orientation from poses
    # Extract position and orientation from poses
    positions = poses[start_idx:start_idx + num_samples, [3, 7, 11]]
    rotation_matrices = poses[start_idx:start_idx + num_samples, :9].reshape(-1, 3, 3)
    orientations = Rotation.from_matrix(rotation_matrices).as_euler('xyz', degrees=True)

    # Plot pose data
    pose_time = np.arange(num_samples) / 10  # Assuming poses are at 10 Hz
    pos_labels = ['x', 'y', 'z']
    pos_colors = ['r', 'g', 'b']
    for i in range(3):
        axes[i + 6].plot(pose_time, positions[:, i], color=pos_colors[i])
        axes[i + 6].set_xlabel('Time (s)', fontsize=8)
        axes[i + 6].set_ylabel(f'Position {pos_labels[i]} (m)', fontsize=8)
        axes[i + 6].set_title(f'Position {pos_labels[i]}', fontsize=10)
        axes[i + 6].tick_params(axis='both', which='major', labelsize=8)
        axes[i + 6].set_ylim(pos_ranges[i])

    # Plot orientation
    ori_labels = ['roll', 'pitch', 'yaw']
    ori_colors = ['c', 'm', 'y']
    for i in range(3):
        axes[i + 9].plot(pose_time, orientations[:, i], color=ori_colors[i])
        axes[i + 9].set_xlabel('Time (s)', fontsize=8)
        axes[i + 9].set_ylabel(f'Orientation {ori_labels[i]} (degrees)', fontsize=8)
        axes[i + 9].set_title(f'Orientation {ori_labels[i]}', fontsize=10)
        axes[i + 9].tick_params(axis='both', which='major', labelsize=8)
        axes[i + 9].set_ylim(ori_ranges[i])

    # Display start image with heading arrow
    start_img = Image.open(image_files[start_idx])
    start_img_plot = ax_start_img.imshow(start_img)
    ax_start_img.axis('off')
    ax_start_img.set_title(f"Start Image at t={extract_timestamp(image_files[start_idx]):.6f}s", fontsize=10)

    # Display end image with heading arrow
    end_img = Image.open(image_files[start_idx + num_samples - 1])
    end_img_plot = ax_end_img.imshow(end_img)
    ax_end_img.axis('off')
    ax_end_img.set_title(f"End Image at t={extract_timestamp(image_files[start_idx + num_samples - 1]):.6f}s",
                         fontsize=10)

    # Add heading arrows
    start_heading = calculate_heading(poses[start_idx:start_idx + 1])[0]
    end_heading = calculate_heading(poses[start_idx + num_samples - 1:start_idx + num_samples])[0]

    add_heading_arrow(ax_start_img, start_heading)
    add_heading_arrow(ax_end_img, end_heading)

    # Plot AHRS data
    ahrs_labels = ['roll', 'pitch', 'yaw']
    ahrs_colors = ['r', 'g', 'b']
    # Plot AHRS data
    ahrs_labels = ['roll', 'pitch', 'yaw']
    ahrs_colors = ['r', 'g', 'b']
    for i in range(3):
        axes[i + 12].plot(pose_time, ahrs_data[start_idx:start_idx + num_samples, i], color=ahrs_colors[i])
        axes[i + 12].set_xlabel('Time (s)', fontsize=8)
        axes[i + 12].set_ylabel(f'AHRS {ahrs_labels[i]} (degrees)', fontsize=8)
        axes[i + 12].set_title(f'AHRS {ahrs_labels[i]}', fontsize=10)
        axes[i + 12].tick_params(axis='both', which='major', labelsize=8)
        if i < 2:  # Roll and Pitch
            axes[i + 12].set_ylim((-10, 10))
        else:  # Yaw
            axes[i + 12].set_ylim((-180, 180))

    plt.tight_layout()
    return fig, axes, start_img_plot, end_img_plot


def add_heading_arrow(ax, heading):
    arrow_length = 0.2
    center = (0.5, 0.5)
    dx = arrow_length * np.cos(np.radians(heading))
    dy = arrow_length * np.sin(np.radians(heading))
    ax.arrow(center[0], center[1], dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r', transform=ax.transAxes)


def calculate_heading(poses):
    rotation_matrices = poses[:, :9].reshape(-1, 3, 3)
    heading = np.arctan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])
    heading_degrees = np.degrees(heading)
    return heading_degrees


def plot_2d_trajectory(ax, poses, num_samples):
    # Extract positions from poses
    positions = poses[:, [3, 11]]  # X and Z coordinates

    ax.plot(positions[:, 0], positions[:, 1], 'b-')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('2D Trajectory (Top View)')

    # Calculate heading
    heading = calculate_heading(poses)

    # Calculate the center of the trajectory
    center_x = (positions[:, 0].min() + positions[:, 0].max()) / 2
    center_y = (positions[:, 1].min() + positions[:, 1].max()) / 2

    # Set a fixed plot size (adjust these values as needed)
    plot_size = 50  # meters

    # Set fixed axis limits
    ax.set_xlim(center_x - plot_size / 2, center_x + plot_size / 2)
    ax.set_ylim(center_y - plot_size / 2, center_y + plot_size / 2)

    # Plot arrows to show heading direction
    arrow_indices = np.linspace(0, len(poses) - 1, 20, dtype=int)
    for i in arrow_indices:
        ax.arrow(positions[i, 0], positions[i, 1],
                 np.cos(np.radians(heading[i])), np.sin(np.radians(heading[i])),
                 head_width=0.5, head_length=1, fc='r', ec='r')

    # Add a small red dot to mark the start point
    ax.plot(positions[0, 0], positions[0, 1], 'ro', markersize=5)


# Add these lines before loading the data
data_dir = "/Users/shlomia/work/my_repo/deeptransvio/data"  # Update this path to your actual data directory
sequence = "00"  # Update this to the sequence number you want to visualize

imu_file = os.path.join(data_dir, "imus", f"{sequence}.mat")
pose_file = os.path.join(data_dir, "poses", f"{sequence}.txt")
image_dir = os.path.join(data_dir, "sequences", sequence, "image_2")

# Now load the data
imu_data = load_imu_data(imu_file)
poses = load_poses(pose_file)
image_files = load_images(image_dir)
print(f"Number of image files loaded: {len(image_files)}")
print(f"First few image files: {image_files[:5]}")

ahrs_data = generate_ahrs_data(poses)

# Calculate data ranges
data_ranges = get_data_ranges(imu_data, poses, ahrs_data)

# Initial plot
start_idx = 0
num_samples = 10  # You can adjust this value
fig, axes, start_img_plot, end_img_plot = plot_sequence(imu_data, poses, image_files, ahrs_data, start_idx, num_samples,
                                                        data_ranges)


def update_plot(fig, axes, start_img_plot, end_img_plot, imu_data, poses, image_files, ahrs_data, start_idx,
                num_samples, data_ranges):
    acc_ranges, gyro_ranges, pos_ranges, ori_ranges, ahrs_ranges = data_ranges

    # Ensure we don't exceed the available data
    num_samples = min(num_samples, len(poses) - start_idx, len(image_files) - start_idx)
    imu_samples = min(num_samples * 10, len(imu_data) - start_idx * 10)

    # Update IMU data
    imu_time = np.arange(imu_samples) / 100  # Assuming 100 Hz IMU data
    for i in range(3):
        axes[i].clear()
        axes[i].plot(imu_time, imu_data[start_idx * 10:start_idx * 10 + imu_samples, i])
        axes[i].set_xlabel('Time (s)', fontsize=8)
        axes[i].set_ylabel(f'Acc {["x", "y", "z"][i]} (m/s^2)', fontsize=8)
        axes[i].set_title(f'Accelerometer {["x", "y", "z"][i]}', fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].set_ylim(acc_ranges[i])

    for i in range(3):
        axes[i + 3].clear()
        axes[i + 3].plot(imu_time, imu_data[start_idx * 10:start_idx * 10 + imu_samples, i + 3])
        axes[i + 3].set_xlabel('Time (s)', fontsize=8)
        axes[i + 3].set_ylabel(f'Gyro {["x", "y", "z"][i]} (rad/s)', fontsize=8)
        axes[i + 3].set_title(f'Gyroscope {["x", "y", "z"][i]}', fontsize=10)
        axes[i + 3].tick_params(axis='both', which='major', labelsize=8)
        axes[i + 3].set_ylim(gyro_ranges[i])

    # Update pose data
    positions = poses[start_idx:start_idx + num_samples, [3, 7, 11]]
    rotation_matrices = poses[start_idx:start_idx + num_samples, :9].reshape(-1, 3, 3)
    orientations = Rotation.from_matrix(rotation_matrices).as_euler('xyz', degrees=True)
    pose_time = np.arange(num_samples) / 10  # Assuming poses are at 10 Hz

    for i in range(3):
        axes[i + 6].clear()
        axes[i + 6].plot(pose_time, positions[:, i])
        axes[i + 6].set_xlabel('Time (s)', fontsize=8)
        axes[i + 6].set_ylabel(f'Position {["x", "y", "z"][i]} (m)', fontsize=8)
        axes[i + 6].set_title(f'Position {["x", "y", "z"][i]}', fontsize=10)
        axes[i + 6].tick_params(axis='both', which='major', labelsize=8)
        axes[i + 6].set_ylim(pos_ranges[i])

    for i in range(3):
        axes[i + 9].clear()
        axes[i + 9].plot(pose_time, orientations[:, i])
        axes[i + 9].set_xlabel('Time (s)', fontsize=8)
        axes[i + 9].set_ylabel(f'Orientation {["roll", "pitch", "yaw"][i]} (degrees)', fontsize=8)
        axes[i + 9].set_title(f'Orientation {["roll", "pitch", "yaw"][i]}', fontsize=10)
        axes[i + 9].tick_params(axis='both', which='major', labelsize=8)
        axes[i + 9].set_ylim(ori_ranges[i])

    # Update AHRS data
    for i in range(3):
        axes[i + 12].clear()
        axes[i + 12].plot(pose_time, ahrs_data[start_idx:start_idx + num_samples, i])
        axes[i + 12].set_xlabel('Time (s)', fontsize=8)
        axes[i + 12].set_ylabel(f'AHRS {["roll", "pitch", "yaw"][i]} (degrees)', fontsize=8)
        axes[i + 12].set_title(f'AHRS {["roll", "pitch", "yaw"][i]}', fontsize=10)
        axes[i + 12].tick_params(axis='both', which='major', labelsize=8)
        if i < 2:  # Roll and Pitch
            axes[i + 12].set_ylim((-10, 10))
        else:  # Yaw
            axes[i + 12].set_ylim((-180, 180))

    # Update 2D trajectory
    axes[15].clear()
    plot_2d_trajectory(axes[15], poses[start_idx:start_idx + num_samples], num_samples)

    # Update images
    # Update images and heading arrows
    try:
        start_img_path = image_files[start_idx]
        end_img_path = image_files[start_idx + num_samples - 1]

        start_img = Image.open(start_img_path)
        end_img = Image.open(end_img_path)

        start_img_plot.set_data(start_img)
        end_img_plot.set_data(end_img)

        axes[-2].set_title(f"Start Image at t={extract_timestamp(start_img_path):.6f}s", fontsize=10)
        axes[-1].set_title(f"End Image at t={extract_timestamp(end_img_path):.6f}s", fontsize=10)

        # Update heading arrows
        start_heading = calculate_heading(poses[start_idx:start_idx + 1])[0]
        end_heading = calculate_heading(poses[start_idx + num_samples - 1:start_idx + num_samples])[0]

        axes[-2].clear()
        axes[-2].imshow(start_img)
        add_heading_arrow(axes[-2], start_heading)
        axes[-2].axis('off')
        axes[-2].set_title(f"Start Image at t={extract_timestamp(start_img_path):.6f}s", fontsize=10)

        axes[-1].clear()
        axes[-1].imshow(end_img)
        add_heading_arrow(axes[-1], end_heading)
        axes[-1].axis('off')
        axes[-1].set_title(f"End Image at t={extract_timestamp(end_img_path):.6f}s", fontsize=10)

    except Exception as e:
        print(f"Error loading images: {str(e)}")
        # If image loading fails, display a blank (black) image
        blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
        start_img_plot.set_data(blank_img)
        end_img_plot.set_data(blank_img)
        axes[-2].set_title("Start Image (Failed to load)", fontsize=10)
        axes[-1].set_title("End Image (Failed to load)", fontsize=10)

    plt.tight_layout()
    fig.canvas.draw_idle()


class Index:
    def __init__(self, start_idx, num_samples, max_idx, data_ranges):
        self.start_idx = start_idx
        self.num_samples = num_samples
        self.max_idx = max_idx
        self.data_ranges = data_ranges

    def next(self, event):
        self.start_idx = min(self.start_idx + self.num_samples, self.max_idx - self.num_samples)
        update_plot(fig, axes, start_img_plot, end_img_plot, imu_data, poses, image_files, ahrs_data, self.start_idx,
                    self.num_samples, self.data_ranges)

    def prev(self, event):
        self.start_idx = max(0, self.start_idx - self.num_samples)
        update_plot(fig, axes, start_img_plot, end_img_plot, imu_data, poses, image_files, ahrs_data, self.start_idx,
                    self.num_samples, self.data_ranges)


callback = Index(start_idx, num_samples, min(len(imu_data), len(poses), len(image_files)), data_ranges)
fig.subplots_adjust(bottom=0.1)  # Adjust the bottom margin to make room for buttons

button_width = 0.06
button_height = 0.03
button_gap = 0.01

prev_button_left = 0.5 - button_width - button_gap / 2
next_button_left = 0.5 + button_gap / 2

ax_prev = plt.axes(tuple([prev_button_left, 0.02, button_width, button_height]))
ax_next = plt.axes(tuple([next_button_left, 0.02, button_width, button_height]))

bnext = Button(ax_next, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(ax_prev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()