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
from utils.utils import *
from tqdm import tqdm 

class data_partition():
    def __init__(self, opt, folder):
        super(data_partition, self).__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len
        self.folder = folder
        self.load_data()

    
    def load_data(self):
        image_dir = self.data_dir + '/sequences/'
        imu_dir = self.data_dir + '/imus/'
        pose_dir = self.data_dir + '/poses/'
    
        # Read all poses
        all_poses, all_poses_rel = read_pose_from_text(f'{pose_dir}qualisys_ariel_odom_traj_3_id1.kitti')
    
        # Count images in each folder and determine pose segments
        pose_start = 0
        for seq in range(1):  # 0 to 6
            img_paths = glob.glob(f'{image_dir}cam0_{seq}/*.png')
            img_count = len(img_paths)
            
            if seq == self.folder:
                self.img_paths = img_paths
                self.poses = all_poses[pose_start:pose_start + img_count]
                self.poses_rel = all_poses_rel[pose_start:pose_start + img_count - 1]
            
            pose_start += img_count
    
        # Sort image paths
        self.img_paths.sort()
    
        # Load IMU data
        self.imus = np.loadtxt(f'{imu_dir}imu_data_{self.folder}.txt', usecols=range(1, 7))
    
        # Create lists for batching
        self.img_paths_list, self.poses_list, self.imus_list = [], [], []
        start = 0
        n_frames = len(self.img_paths)
        while start + self.seq_len < n_frames:
            self.img_paths_list.append(self.img_paths[start:start + self.seq_len])
            self.poses_list.append(self.poses_rel[start:start + self.seq_len - 1])
            self.imus_list.append(self.imus[start * 10:(start + self.seq_len - 1) * 10 + 1])
            start += self.seq_len - 1
        self.img_paths_list.append(self.img_paths[start:])
        self.poses_list.append(self.poses_rel[start:])
        self.imus_list.append(self.imus[start * 10:])
    
        print(f"Loaded {len(self.img_paths)} images, {len(self.poses)} poses, and {len(self.imus)} IMU readings for sequence {self.folder}")

    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, i):
        image_path_sequence = self.img_paths_list[i]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.opt.img_h, self.opt.img_w))
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        imu_sequence = torch.FloatTensor(self.imus_list[i])
        gt_sequence = self.poses_list[i][:, :6]
        return image_sequence, imu_sequence, gt_sequence


class NTNU_tester():
    def __init__(self, args):
        super(NTNU_tester, self).__init__()
        
        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))

        self.args = args

    def generate_extended_plots(self, result_dir):
        for i, seq in enumerate(self.args.val_seq):
            pose_est_global = self.est[i]['pose_est_global']
            pose_gt_global = self.est[i]['pose_gt_global']
            plotExtendedAnalysis(seq, pose_gt_global, pose_est_global, result_dir)

    def test_one_path(self, net, df, selection, num_gpu=1, p=0.5):
        hc = None
        pose_list, decision_list, probs_list= [], [], []
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            image_seq = image_seq.repeat(1, 3, 1, 1)
            x_in = image_seq.unsqueeze(0).repeat(num_gpu,1,1,1,1).cuda()
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu,1,1).cuda()
            with torch.no_grad():
                pose, hc = net(x_in, i_in, is_first=(i==0), hc=hc, selection=selection, p=p)
            pose_list.append(pose[0,:,:].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], selection, num_gpu=num_gpu, p=p)
            
            # Crop the relevant part of the ground truth poses
            total_poses = len(self.dataloader[i].poses_rel)
            if seq == 6:
                start_index = total_poses - 838  # Use the last 838 poses for sequence 6
            else:
                # For other sequences, you might need to adjust this logic
                # For now, let's assume we use all poses for other sequences
                start_index = 0
            
            # Add end_index based on the length of pose_est
            end_index = start_index + len(pose_est)
            
            cropped_poses_rel = self.dataloader[i].poses_rel[start_index:end_index]
            
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = ntnu_eval(pose_est, cropped_poses_rel)
            
            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'speed':speed})
            self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse})
        
        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D_and_3D(seq,
                        self.est[i]['pose_gt_global'], 
                        self.est[i]['pose_est_global'], 
                        save_dir, 
                        self.est[i]['speed'],
                        window_size)
    
    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir/'{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))


def ntnu_eval(pose_est, pose_gt):
    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    # Transfer to 3x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    # Using KITTI metric (which is also applicable for NTNU dataset)
    err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)
    
    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, speed

def plotExtendedAnalysis(seq, poses_gt_mat, poses_est_mat, plot_path_dir):
    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]

    # Extract x, y, z coordinates
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), dpi=100)

    # 1. Scale difference plot (top subplot)
    path_length_gt = np.cumsum(np.sqrt(np.diff(x_gt)**2 + np.diff(y_gt)**2 + np.diff(z_gt)**2))
    path_length_pred = np.cumsum(np.sqrt(np.diff(x_pred)**2 + np.diff(y_pred)**2 + np.diff(z_pred)**2))
    ax1.plot(path_length_gt, label='Ground Truth')
    ax1.plot(path_length_pred, label='Prediction')
    ax1.set_xlabel('Step', fontsize=fontsize_)
    ax1.set_ylabel('Cumulative Path Length (m)', fontsize=fontsize_)
    ax1.legend(loc="upper left", prop={'size': fontsize_})
    ax1.set_title('Scale Comparison')

    # 2. Y-axis error plot (middle subplot)
    y_error = y_pred - y_gt
    ax2.plot(y_error, label='Y-axis Error')
    ax2.set_xlabel('Step', fontsize=fontsize_)
    ax2.set_ylabel('Y-axis Error (m)', fontsize=fontsize_)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.legend(loc="upper right", prop={'size': fontsize_})
    ax2.set_title('Y-axis Error')

    # 3. 3D error magnitude plot (bottom subplot)
    error_3d = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2 + (z_pred - z_gt)**2)
    ax3.plot(error_3d, label='3D Error Magnitude')
    ax3.set_xlabel('Step', fontsize=fontsize_)
    ax3.set_ylabel('3D Error Magnitude (m)', fontsize=fontsize_)
    ax3.legend(loc="upper right", prop={'size': fontsize_})
    ax3.set_title('3D Error Magnitude')

    plt.tight_layout()
    png_title = "{}_extended_analysis".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Additional plot: Cumulative error
    plt.figure(figsize=(10, 5))
    cumulative_error = np.cumsum(error_3d)
    plt.plot(cumulative_error, label='Cumulative 3D Error')
    plt.xlabel('Step', fontsize=fontsize_)
    plt.ylabel('Cumulative 3D Error (m)', fontsize=fontsize_)
    plt.title('Cumulative 3D Error Over Time')
    plt.legend(loc="upper left", prop={'size': fontsize_})
    png_title = "{}_cumulative_error".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()



def kitti_err_cal(pose_est_mat, pose_gt_mat):

    lengths = [1, 2, 3, 4, 5, 6, 7, 8]
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):

        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, len_)
            # Continue if sequence not long enough
            if last_frame == -1 or last_frame >= len(pose_est_mat) or first_frame >= len(pose_est_mat):
                continue

            pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame])

            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / len_, t_err / len_, len_])

    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotPath_2D_and_3D(seq, poses_gt_mat, poses_est_mat, plot_path_dir, speed, window_size):
    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]
    start_point = [0, 0, 0]
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    # Plot 2D trajectory estimation map
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_gt, y_gt, z_gt, style_gt, label=plot_keys[0])
    ax.plot(x_pred, y_pred, z_pred, style_pred, label=plot_keys[1])
    ax.scatter(start_point[0], start_point[1], start_point[2], c='k', marker='o', s=50, label='Start Point')
    ax.legend(loc="upper right", prop={'size': fontsize_})
    ax.set_xlabel('X (m)', fontsize=fontsize_)
    ax.set_ylabel('Y (m)', fontsize=fontsize_)
    ax.set_zlabel('Z (m)', fontsize=fontsize_)
    ax.set_title('3D Trajectory')

    # Ensure equal aspect ratio
    max_range = np.array([x_gt.max()-x_gt.min(), y_gt.max()-y_gt.min(), z_gt.max()-z_gt.min()]).max() / 2.0
    mid_x = (x_gt.max()+x_gt.min()) * 0.5
    mid_y = (y_gt.max()+y_gt.min()) * 0.5
    mid_z = (z_gt.max()+z_gt.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    png_title = "{}_path_3d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot the speed map (unchanged)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = speed
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_speed = max(cout)
    min_speed = min(cout)
    ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + 'm/s' for i in ticks])

    plt.title('speed heatmap')
    png_title = "{}_speed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

