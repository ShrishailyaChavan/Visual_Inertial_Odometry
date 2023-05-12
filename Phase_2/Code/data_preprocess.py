import os
import glob
import pandas as pd
import numpy as np
import cv2
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.spatial.transform import Rotation as R


class EuRoCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # self.image_files = sorted(glob.glob(os.path.join(root_dir, 'mav0', 'cam0', 'data', '*.png')))
        # Read image files from both 'cam0' and 'cam1' folders
        cam0_image_files = sorted(glob.glob(os.path.join(root_dir, 'mav0', 'cam0', 'data', '*.png')))
        cam1_image_files = sorted(glob.glob(os.path.join(root_dir, 'mav0', 'cam1', 'data', '*.png')))

        # Combine image files from both cameras
        self.image_files = cam0_image_files + cam1_image_files
        self.imu_data = pd.read_csv(os.path.join(root_dir, 'mav0', 'imu0', 'data.csv'))
        self.ground_truth = pd.read_csv(os.path.join(root_dir, 'mav0', 'state_groundtruth_estimate0', 'data.csv'))

    def __len__(self):
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        img1 = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.image_files[idx + 1], cv2.IMREAD_GRAYSCALE)

        # Check if img1 and img2 are read correctly
        if img1 is None or img2 is None:
            print(f"Failed to read image(s) at index {idx}")
            return None, None, None, None


        # Reshape images to (C, H, W) and normalize
        img1 = np.expand_dims(img1, axis=0) / 255.0
        img2 = np.expand_dims(img2, axis=0) / 255.0

        # Convert img1 and img2 to float tensors
        img1 = torch.tensor(img1, dtype=torch.float32)
        img2 = torch.tensor(img2, dtype=torch.float32)

        # # Reshape images to (C, H, W) and normalize
        # img1 = np.stack((img1,) * 3, axis=-1) / 255.0
        # img2 = np.stack((img2,) * 3, axis=-1) / 255.0

        # # Convert img1 and img2 to float tensors
        # img1 = torch.tensor(img1.transpose(2, 0, 1), dtype=torch.float32)
        # img2 = torch.tensor(img2.transpose(2, 0, 1), dtype=torch.float32)

        # Get timestamps of images and find IMU measurements between them
        img1_timestamp = int(os.path.basename(self.image_files[idx]).split('.')[0])
        img2_timestamp = int(os.path.basename(self.image_files[idx + 1]).split('.')[0])

        imu_seq = self.imu_data[(self.imu_data['#timestamp [ns]'] >= img1_timestamp) & (self.imu_data['#timestamp [ns]'] <= img2_timestamp)]
        imu_seq = imu_seq.iloc[:, 1:].values

        # Convert imu_seq to float tensor
        imu_seq = torch.tensor(imu_seq, dtype=torch.float32)

        # Get ground truth pose for img1 and img2
        gt_pose1 = self.ground_truth[self.ground_truth['#timestamp'] == img1_timestamp]
        gt_pose2 = self.ground_truth[self.ground_truth['#timestamp'] == img2_timestamp]

        if gt_pose1.empty or gt_pose2.empty:
            # print("No matching ground truth data found for img1_timestamp:", img1_timestamp, "or img2_timestamp:", img2_timestamp)
            return None

        gt_pose1 = gt_pose1.iloc[0, 1:8].values  # Extract position and orientation only
        gt_pose2 = gt_pose2.iloc[0, 1:8].values  # Extract position and orientation only

        # Calculate relative pose between img1 and img2
        gt_rel_position = gt_pose2[:3] - gt_pose1[:3]

        # Calculate relative orientation
        quat1_inv = R.from_quat(gt_pose1[3:]).inv()
        quat_rel = quat1_inv * R.from_quat(gt_pose2[3:])
        gt_rel_orientation = quat_rel.as_quat()

        gt_rel_pose = np.hstack((gt_rel_position, gt_rel_orientation))

        # Convert gt_rel_pose to float tensor
        gt_rel_pose = torch.tensor(gt_rel_pose, dtype=torch.float32)

        if self.transform:
            img1, img2, imu_seq, gt_rel_pose = self.transform((img1, img2, imu_seq, gt_rel_pose))

        return img1, img2, imu_seq, gt_rel_pose


# Create a dataset instance for each sequence
datasets = []
# for seq in ['MH_01_easy','MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult']:
#     datasets.append(EuRoCDataset(os.path.join('/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/Data', seq)))

for seq in ['MH_01_easy']:
    datasets.append(EuRoCDataset(os.path.join('/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/Data', seq)))

# Concatenate all datasets
full_dataset = ConcatDataset(datasets)

# Split into training, validation, and test sets
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
    
