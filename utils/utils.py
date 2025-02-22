import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
plt.switch_backend('agg')

_EPS = np.finfo(float).eps * 4.0

def isRotationMatrix(R):
    '''
    check whether a matrix is a qualified rotation metrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_relative_pose(Rt1, Rt2):
    '''
    Calculate the relative 4x4 pose matrix between two pose matrices
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel

def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    Calculate the relative rotation and translation from two consecutive pose matrices 
    '''
    
    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

def rotationError(Rt1, Rt2):
    '''
    Calculate the rotation difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(Rt1, Rt2):
    '''
    Calculate the translational difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def eulerAnglesToRotationMatrix(theta):
    '''
    Calculate the rotation matrix from eular angles (roll, yaw, pitch)
    '''
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def normalize_angle_delta(angle):
    '''
    Normalization angles to constrain that it is between -pi and pi
    '''
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def pose_6DoF_to_matrix(pose):
    '''
    Calculate the 3x4 transformation matrix from Eular angles and translation vector
    '''
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)
    R = np.concatenate((R, t), 1)
    R = np.concatenate((R, np.array([[0, 0, 0, 1]])), 0)
    return R

def pose_accu(Rt_pre, R_rel):
    '''
    Calculate the accumulated pose from the latest pose and the relative rotation and translation
    '''
    Rt_rel = pose_6DoF_to_matrix(R_rel)
    return Rt_pre @ Rt_rel

def path_accu(pose):
    '''
    Generate the global pose matrices from a series of relative poses
    '''
    answer = [np.eye(4)]
    for index in range(pose.shape[0]):
        pose_ = pose_accu(answer[-1], pose[index, :])
        answer.append(pose_)
    return answer

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def rmse_err_cal(pose_est, pose_gt):
    '''
    Calculate the rmse of relative translation and rotation
    '''
    t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
    return t_rmse, r_rmse

def trajectoryDistances(poses):
    '''
    Calculate the distance and speed for each frame
    '''
    dist = [0]
    speed = [0]
    for i in range(len(poses) - 1):
        cur_frame_idx = i
        next_frame_idx = cur_frame_idx + 1
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        speed.append(np.sqrt(dx**2 + dy**2 + dz**2) * 10)
    return dist, speed

def lastFrameFromSegmentLength(dist, first_frame, len_):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + len_):
            return i
    return -1

def computeOverallErr(seq_err):
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)

    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err

def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values= np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt
    
def read_pose_from_text(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        values_p = read_pose(lines[0])
        poses_abs.append(values_p)            
        for i in range(1, len(lines)):
            values = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(values_p, values)) 
            values_p = values.copy()
            poses_abs.append(values) 
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
        
    return poses_abs, poses_rel

def saveSequence(poses, file_name):
    with open(file_name, 'w') as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            f.write(' '.join([str(r) for r in pose]))
            f.write('\n')



import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def visualize_imu_sequence_current_processing(imu_data, save_path=None):
    # Assuming imu_data shape is (N, seq_len, 11, 6)
    # We'll visualize the first sample in the batch
    imu_data = imu_data[0]  # Shape: (seq_len, 11, 6)

    # Create a single plot for all 6 channels
    fig, ax = plt.subplots(figsize=(12, 8))

    channel_names = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for seq in range(imu_data.shape[0]):
        for channel in range(6):
            ax.plot(range(11), imu_data[seq, :, channel],
                    color=colors[channel], alpha=0.5,
                    label=f'{channel_names[channel]} (Seq {seq})' if seq == 0 else "")

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Sensor Value')
    ax.set_title('IMU Sequence Visualization (All Channels)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        # Add timestamp to the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, ext = os.path.splitext(save_path)
        save_path = f"{filename}_{timestamp}{ext}"
        plt.savefig(save_path)
        plt.close()
        print(f"IMU sequence visualization saved to {save_path}")
    else:
        plt.show()


def visualize_imu_sequence(imu_data, save_path=None):
    # Assuming imu_data shape is (N, seq_len, 11, 6)
    # We'll visualize the first sample in the batch
    imu_data = imu_data[0]  # Shape: (seq_len, 11, 6)
    seq_len = imu_data.shape[0]

    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('IMU Sequence Visualization', fontsize=16)

    sensor_names = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z',
                    'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']

    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axs[row, col]

        # Extract data for each sensor
        sensor_data = imu_data[:, :, i].T  # Shape: (11, seq_len)

        im = ax.imshow(sensor_data, aspect='auto', cmap='coolwarm')
        ax.set_title(sensor_names[i])
        ax.set_ylabel('IMU Readings')
        ax.set_xlabel('Frame Number')

        ax.set_yticks(range(11))
        ax.set_xticks(np.arange(0, seq_len, seq_len // 10))
        ax.set_xticklabels(range(0, seq_len, seq_len // 10))

        plt.colorbar(im, ax=ax)

    plt.tight_layout()

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

        # Save the figure
        plt.savefig(full_save_path)
        plt.close()
        print(f"Figure saved as: {full_save_path}")
    else:
        plt.show()

# Add this function to the existing utils.py file