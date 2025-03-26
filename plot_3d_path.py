import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            # Read 12 values per line
            pose = np.array(list(map(float, line.split())))
            poses.append(pose.reshape(3, 4))  # Reshape to 3x4 matrix
    return poses

def extract_positions(poses):
    return np.array([pose[:3, 3] for pose in poses])

def plot_3d_path(positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    ax.plot(x, y, z, label='Path')
    ax.scatter(x[0], y[0], z[0], c='r', marker='o', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='g', marker='o', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Path')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    file_path = "/home/ws1/work/Shlomi/deeptransvio/data/poses/00.txt"
    poses = read_poses(file_path)
    positions = extract_positions(poses)
    plot_3d_path(positions)