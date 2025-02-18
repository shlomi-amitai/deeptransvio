import sys
sys.path.append('..')
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from path import Path
from utils.utils import rotationError, get_relative_pose_6DoF

from utils import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import pandas as pd
IMU_FREQ = 10


class Aqua(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=[0, 1, 2, 4, 6, 8, 9],
                 transform=None):

        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for seq in self.train_seqs:
            # Read poses from text file
            pose_file = self.root / f'poses/new_archaeo_colmap_traj_sequence_{seq:02d}.txt'
            with open(pose_file, 'r') as f:
                poses_raw = [line.strip().split() for line in f]
    
            # Convert poses to 4x4 matrices
            poses = []
            timestamps = []
            for pose in poses_raw:
                img_num, tx, ty, tz, qx, qy, qz, qw = map(float, pose)
                rotation = Rotation.from_quat([qx, qy, qz, qw])
                translation = np.array([tx, ty, tz])
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = rotation.as_matrix()
                pose_matrix[:3, 3] = translation
                poses.append(pose_matrix)
                timestamps.append(img_num)

            # Interpolate poses
            interpolated_poses = []
            for i in range(len(timestamps) - 1):
                start_time, end_time = timestamps[i], timestamps[i+1]
                start_pose, end_pose = poses[i], poses[i+1]
                
                # Create interpolation times
                interp_times = np.arange(start_time, end_time)
                
                # Interpolate translation
                start_trans = start_pose[:3, 3]
                end_trans = end_pose[:3, 3]
                trans_interp = interp1d([start_time, end_time], np.vstack((start_trans, end_trans)).T)
                
                # Interpolate rotation
                start_rot = Rotation.from_matrix(start_pose[:3, :3])
                end_rot = Rotation.from_matrix(end_pose[:3, :3])
                key_rots = Rotation.from_matrix(np.stack((start_pose[:3, :3], end_pose[:3, :3])))
                key_times = [start_time, end_time]
                slerp = Slerp(key_times, key_rots)
                
                # Generate interpolated poses
                for t in interp_times:
                    inter_trans = trans_interp(t)
                    inter_rot = slerp(t)
                    
                    inter_pose = np.eye(4)
                    inter_pose[:3, :3] = inter_rot.as_matrix()
                    inter_pose[:3, 3] = inter_trans
                    
                    interpolated_poses.append(inter_pose)

            # Add the last pose
            interpolated_poses.append(poses[-1])

            # Replace the original poses with interpolated poses
            poses = interpolated_poses
    
            # Calculate relative poses
            poses_rel = [np.eye(4)]  # First relative pose is identity
            for i in range(1, len(poses)):
                poses_rel.append(np.linalg.inv(poses[i-1]) @ poses[i])

            imu_file = self.root / f'imus/imu_sequence_{seq:01d}.csv'
            imus = pd.read_csv(imu_file, header=None)
            imus = imus.iloc[1:, 1:].astype(float).values  # Convert to float and then to numpy array
            fpaths = sorted((self.root / f'sequences/images_sequence_{seq:01d}/').files("*.png"))
            for i in range(len(fpaths) - self.sequence_length):
                img_samples = fpaths[i:i + self.sequence_length]
                imu_samples = imus[i * IMU_FREQ:(i + self.sequence_length - 1) * IMU_FREQ + 1]
                pose_samples = poses[i:i + self.sequence_length]
                pose_rel_samples = poses_rel[i+1:i + self.sequence_length]  # Start from i+1 to match the number of relative poses
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                
                # Convert pose_rel_samples to 12-parameter format
                pose_rel_12param = [pose_mat[:3, :].flatten() for pose_mat in pose_rel_samples]
                
                sample = {'imgs': img_samples, 'imus': imu_samples, 'gts': pose_rel_12param, 'rot': segment_rot}
                sequence_set.append(sample)
        self.samples = sequence_set
    
        # The rest of the method remains the same
        # Generate weights based on the rotation of the training segments
        rot_list = np.array([np.cbrt(item['rot'] * 180 / np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range) + 1)]
    
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    
        self.weights = [np.float32(1 / eff_label_dist[bin_idx - 1]) for bin_idx in indexes]

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]

        if self.transform is not None:
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)

        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return imgs, imus, gts, rot, weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{:02d} '.format(seq)
        fmt_str += '\n'
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def main():
    # Set up the root directory and other parameters
    root_dir = Path('/Users/shlomia/work/my_repo/deeptransvio/aqua_data/')  # Replace with the actual path to your Aqua dataset
    sequence_length = 11
    train_seqs = [1]

    # Create a transform
    transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Resize((376, 1241))
    ])

    # Initialize the Aqua dataset
    aqua_dataset = Aqua(root=root_dir, 
                        sequence_length=sequence_length, 
                        train_seqs=train_seqs, 
                        transform=transform)

    print(aqua_dataset)
    print(f"Dataset size: {len(aqua_dataset)}")

    # Get a sample from the dataset
    sample_idx = 0
    imgs, imus, gts, rot, weight = aqua_dataset[sample_idx]

    print(f"\nSample {sample_idx}:")
    print(f"Images shape: {imgs[0].shape}")  # Assuming imgs is a list of tensors
    print(f"IMUs shape: {imus.shape}")
    print(f"Ground truths shape: {gts.shape}")
    print(f"Rotation: {rot}")
    print(f"Weight: {weight}")

    # Create a DataLoader
    from torch.utils.data import DataLoader
    batch_size = 4
    dataloader = DataLoader(aqua_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Iterate through a batch
    for batch_idx, (batch_imgs, batch_imus, batch_gts, batch_rots, batch_weights) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Batch images shape: {batch_imgs[0].shape}")  # Assuming batch_imgs is a list of tensors
        print(f"Batch IMUs shape: {batch_imus.shape}")
        print(f"Batch ground truths shape: {batch_gts.shape}")
        print(f"Batch rotations shape: {batch_rots.shape}")
        print(f"Batch weights shape: {batch_weights.shape}")
        break  # We'll just look at the first batch

if __name__ == "__main__":
    main()


