import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from utils.utils import rotationError, translationError, path_accu, rmse_err_cal
from torch.utils.data import Dataset, DataLoader
from path import Path
import pandas as pd
from scipy.spatial.transform import Rotation
from utils.utils import get_relative_pose_6DoF

class Aqua_tester():
    def __init__(self, args):
        super(Aqua_tester, self).__init__()
        
        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))

        self.args = args

    # ... (rest of the class implementation remains the same)

    def test_one_path(self, net, df, selection, num_gpu=1, p=0.5):
        hc = None
        pose_list = []
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):  
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
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse = aqua_eval(pose_est, self.dataloader[i].poses_rel)
            
            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global})
            self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse})
            
        return self.errors

    def generate_plots(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_3D(seq, 
                        self.est[i]['pose_gt_global'], 
                        self.est[i]['pose_est_global'], 
                        save_dir)
    
    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir/'{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))

def aqua_eval(pose_est, pose_gt):
    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    # Transfer to 4x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    # Calculate relative errors
    t_rel, r_rel = relative_error_calc(pose_est_mat, pose_gt_mat)
    
    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse

def relative_error_calc(pose_est_mat, pose_gt_mat):
    t_rel = 0
    r_rel = 0
    step_size = 10  # Adjust as needed

    for i in range(0, len(pose_gt_mat) - step_size, step_size):
        pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[i]), pose_gt_mat[i + step_size])
        pose_delta_est = np.dot(np.linalg.inv(pose_est_mat[i]), pose_est_mat[i + step_size])

        r_err = rotationError(pose_delta_est, pose_delta_gt)
        t_err = translationError(pose_delta_est, pose_delta_gt)

        t_rel += t_err / np.linalg.norm(pose_delta_gt[:3, 3])
        r_rel += r_err

    t_rel /= (len(pose_gt_mat) - step_size) / step_size
    r_rel /= (len(pose_gt_mat) - step_size) / step_size

    return t_rel, r_rel

def plotPath_3D(seq, poses_gt_mat, poses_est_mat, plot_path_dir):
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

    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_gt, y_gt, z_gt, style_gt, label=plot_keys[0])
    ax.plot(x_pred, y_pred, z_pred, style_pred, label=plot_keys[1])
    ax.plot([start_point[0]], [start_point[1]], [start_point[2]], style_O, label='Start Point')
    ax.legend(loc="upper right", prop={'size': fontsize_})
    ax.set_xlabel('X (m)', fontsize=fontsize_)
    ax.set_ylabel('Y (m)', fontsize=fontsize_)
    ax.set_zlabel('Z (m)', fontsize=fontsize_)

    plt.title('3D path')
    png_title = "{}_path_3d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

def saveSequence(poses, filename):
    with open(filename, 'w') as f:
        for pose in poses:
            f.write(' '.join([str(v) for v in pose[:3, :].flatten()]) + '\n')



class AquaSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence, transform=None):
        self.root_dir = Path(root_dir)
        self.sequence = sequence
        self.transform = transform
        
        self.image_dir = self.root_dir / 'sequences' / f'images_sequence_{sequence:01d}'
        self.imu_file = self.root_dir / 'imus' / f'imu_sequence_{sequence:01d}.csv'
        self.pose_file = self.root_dir / 'poses' / f'new_archaeo_colmap_traj_sequence_{sequence:02d}.txt'
        
        self.images = sorted(self.image_dir.files('*.png'))
        self.imus = pd.read_csv(self.imu_file, header=None)
        self.imus = self.imus.iloc[1:, 1:].astype(float).values
        
        self.poses = self.load_poses()
        self.poses_rel = self.get_relative_poses()

    def load_poses(self):
        poses = []
        with open(self.pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                t = values[1:4]
                q = values[4:]  # [qx, qy, qz, qw]
                R = Rotation.from_quat(q).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                poses.append(T)
        return poses

    def get_relative_poses(self):
        poses_rel = []
        for i in range(1, len(self.poses)):
            rel_pose = get_relative_pose_6DoF(self.poses[i-1], self.poses[i])
            poses_rel.append(rel_pose)
        return poses_rel

    def __len__(self):
        return len(self.images) - 1  # We need pairs of images

    def __getitem__(self, idx):
        img1_path = self.images[idx]
        img2_path = self.images[idx + 1]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Stack images to match the expected format
        imgs = torch.stack([img1, img2], 0)
        
        imu = self.imus[idx * 10:(idx + 1) * 10 + 1]  # Assuming 10Hz IMU data between frames
        imu = torch.from_numpy(imu).float()
        
        pose = self.poses_rel[idx]
        pose = torch.from_numpy(pose).float()

        return imgs, imu, pose

def data_partition(args, sequence):
    root_dir = Path('./aqua_data/')
    
    dataset = AquaSequenceDataset(root_dir, sequence, transform=args.transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    return dataloader