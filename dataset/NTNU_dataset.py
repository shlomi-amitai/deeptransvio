import sys
sys.path.append('..')
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from path import Path
from utils.utils import rotationError, read_pose_from_text, get_relative_pose_6DoF
from utils import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import pandas as pd

IMU_FREQ = 10

class NTNU(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=[0, 1, 2, 3, 4, 5],  # Adjust these sequences as needed
                 transform=None):
        
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()
    
    def make_dataset(self):
        sequence_set = []
        
        # Read all poses
        all_poses, all_poses_rel = read_pose_from_text(self.root/'poses/qualisys_ariel_odom_traj_3_id1.kitti')
        
        pose_start = 0
        for seq in self.train_seqs:
            # Read IMU data from txt file
            imu_file = self.root / f'imus/imu_data_{seq}.txt'
            imus = pd.read_csv(imu_file, header=None, delim_whitespace=True, usecols=range(1, 7)).values
            
            # Get image paths
            fpaths = sorted((self.root/f'sequences/cam0_{seq}').files("*.png"))
            
            # Determine the pose segment for this sequence
            img_count = len(fpaths)
            poses = all_poses[pose_start:pose_start + img_count]
            poses_rel = all_poses_rel[pose_start:pose_start + img_count - 1]
            
            for i in range(len(fpaths) - self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                imu_samples = imus[i*IMU_FREQ:(i+self.sequence_length-1)*IMU_FREQ+1]
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                sample = {'imgs':img_samples, 'imus':imu_samples, 'gts': pose_rel_samples, 'rot': segment_rot}
                sequence_set.append(sample)
            
            pose_start += img_count
        
        self.samples = sequence_set
        
        # Generate weights based on the rotation of the training segments
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]
    
        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    
        self.weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]
    
        print(f"Loaded {len(self.samples)} samples from {len(self.train_seqs)} sequences")

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
            fmt_str += '{} '.format(seq)
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


class NTNUSmallDataset(NTNU):
    def __init__(self, root, sequence_length=11, train_seqs=[1, 2, 3, 4, 5], transform=None, sample_size=1000):
        super().__init__(root, sequence_length, train_seqs, transform)
        self.sample_size = sample_size
        self.subsample_dataset()

    def subsample_dataset(self):
        if len(self.samples) > self.sample_size:
            self.samples = random.sample(self.samples, self.sample_size)
            # Recalculate weights for the subsampled dataset
            self.recalculate_weights()

    def recalculate_weights(self):
        # This method recalculates the weights for the subsampled dataset
        # using the same method as in the original NTNU class
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]

        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        self.weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.sample_size)
        fmt_str += '    ' + super().__repr__()
        return fmt_str