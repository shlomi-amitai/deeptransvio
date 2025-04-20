import sys
sys.path.append('..')
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from path import Path
from utils.utils import rotationError, read_pose_from_text
from utils import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from scipy.spatial.transform import Rotation

IMU_FREQ = 10

class KITTI(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '06', '08', '09'],
                 transform=None):
        
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()
    
    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            poses, poses_rel = read_pose_from_text(self.root/'poses/{}.txt'.format(folder))
            print(f"Shape of poses array for folder {folder}: {poses.shape}")
            imus = sio.loadmat(self.root/'imus/{}.mat'.format(folder))['imu_data_interp']
            fpaths = sorted((self.root/'sequences/{}/image_2'.format(folder)).files("*.png"))
            
            # Extract AHRS data from poses
            ahrs_data = poses_rel[:,:3]
            
            for i in range(len(fpaths)-self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                imu_samples = imus[i*IMU_FREQ:(i+self.sequence_length-1)*IMU_FREQ+1]
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                ahrs_samples = ahrs_data[i:i+self.sequence_length-1]
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                sample = {
                    'imgs': img_samples, 
                    'imus': imu_samples, 
                    'gts': pose_rel_samples, 
                    'ahrs': ahrs_samples,
                    'rot': segment_rot
                }
                sequence_set.append(sample)
        self.samples = sequence_set
        
        # Generate weights based on the rotation of the training segments
        # Weights are calculated based on the histogram of rotations according to the method in https://github.com/YyzHarry/imbalanced-regression
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]

        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        self.weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [Image.open(img) for img in sample['imgs']]
        if self.transform:
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts'])
        
        ahrs = np.copy(sample['ahrs'])  # Add this line to handle AHRS data
        
        # Calculate the weight
        weight = np.linalg.norm(gts[:, :3], axis=1)
        weight[weight < 0.01] = 0.01
        weight = weight / weight.sum()
        
        return imgs, imus, gts, ahrs, weight

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

def extract_ahrs_from_poses(poses):
    # poses is already in the shape (n, 4, 4)
    # Extract the rotation matrix (3x3) from each pose
    rotation_matrices = poses[:, :3, :3]
    
    # Convert rotation matrices to Euler angles (roll, pitch, yaw)
    euler_angles = Rotation.from_matrix(rotation_matrices).as_euler('xyz', degrees=True)
    
    # Add some noise to simulate real AHRS data (optional)
    noise = np.random.normal(0, 0.1, euler_angles.shape)
    noisy_euler_angles = euler_angles + noise
    
    return noisy_euler_angles
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, imus, gts):
        for t in self.transforms:
            images, imus, gts = t(images, imus, gts)
        return images, imus, gts

class ToTensor(object):
    def __call__(self, images, imus, gts):
        # Convert everything to tensors
        images = [transforms.ToTensor()(image) for image in images]
        imus = torch.from_numpy(imus).float()
        gts = torch.from_numpy(gts).float()
        return images, imus, gts

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, imus, gts):
        images = [transforms.Resize(self.size)(image) for image in images]
        return images, imus, gts

# Update other transform classes similarly...