import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from data_preprocess_new import test_dataset, custom_collate_test
from data_preprocess import test_dataset, custom_collate, train_dataset
from network import VisionOnlyNetwork, InertialOnlyNetwork, VisualInertialNetwork
from loss_fn import TranslationRotationLoss, pose_loss
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

# from train import vision_loss_fn, inertial_loss_fn, visual_inertial_loss_fn

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_gt_positions = []
    all_pred_positions = []
    all_gt_rotations = []
    all_pred_rotations = []

    with torch.no_grad():
        for data in dataloader:
            if data is None:
                continue
            img1, img2, imu_data, gt_rel_pose = [d.to(device) for d in data]

            if isinstance(model, VisionOnlyNetwork):
                pred_rel_pose = model(img1, img2)
            elif isinstance(model, InertialOnlyNetwork):
                pred_rel_pose = model(imu_data)
            elif isinstance(model, VisualInertialNetwork):
                pred_rel_pose = model(img1, img2, imu_data)
            else:
                raise ValueError("Unknown model type")

            loss = pose_loss(pred_rel_pose, gt_rel_pose, alpha=0.5)
            total_loss += loss.item() * gt_rel_pose.size(0)
            total_samples += gt_rel_pose.size(0)

            gt_positions, pred_positions, gt_rotations, pred_rotations = integrate_poses(gt_rel_pose, pred_rel_pose, device)
            all_gt_positions.extend(gt_positions)
            all_pred_positions.extend(pred_positions)
            all_gt_rotations.extend(gt_rotations)
            all_pred_rotations.extend(pred_rotations)

    mean_loss = total_loss / total_samples
    return mean_loss, all_gt_positions, all_pred_positions, all_gt_rotations, all_pred_rotations

def integrate_poses(gt_rel_pose, pred_rel_pose, device):
    gt_positions = [torch.zeros((3,), device=device)]  # Initial position at (0, 0, 0)
    pred_positions = [torch.zeros((3,), device=device)]
    gt_rotations = [torch.zeros((4,), device=device)]  # Initial rotation angles (0, 0, 0, 0)
    pred_rotations = [torch.zeros((4,), device=device)]

    for i in range(len(gt_rel_pose)):
        gt_positions.append(gt_positions[-1] + gt_rel_pose[i, :3].to(device))
        pred_positions.append(pred_positions[-1] + pred_rel_pose[i, :3].to(device))
        gt_rotations.append(gt_rotations[-1] + gt_rel_pose[i, 3:].to(device))
        pred_rotations.append(pred_rotations[-1] + pred_rel_pose[i, 3:].to(device))

    return gt_positions, pred_positions, gt_rotations, pred_rotations


def plot_trajectory(gt_positions, pred_positions, title):
    gt_positions = torch.stack(gt_positions).cpu().numpy()
    pred_positions = torch.stack(pred_positions).cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(gt_positions[:, i], label="Ground Truth")
        axs[i].plot(pred_positions[:, i], label="Prediction")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid()

    fig.suptitle(title)
    plt.show()

def plot_rotations(gt_rotations, pred_rotations, title):
    gt_rotations_np = torch.stack(gt_rotations).cpu().numpy()
    pred_rotations_np = torch.stack(pred_rotations).cpu().numpy()

    gt_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in gt_rotations_np if np.linalg.norm(q) != 0])
    pred_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in pred_rotations_np if np.linalg.norm(q) != 0])

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs[i].plot(gt_euler_angles[:, i], label="Ground Truth")
        axs[i].plot(pred_euler_angles[:, i], label="Prediction")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(labels[i] + " (Radians)")
        axs[i].legend()
        axs[i].grid()

    fig.suptitle(title)
    plt.show()


def scatter_trajectory_3d(gt_positions, pred_positions, gt_rotations, pred_rotations, title):
    gt_positions = torch.stack(gt_positions).cpu().numpy()
    pred_positions = torch.stack(pred_positions).cpu().numpy()
    gt_rotations = torch.stack(gt_rotations).cpu().numpy()
    pred_rotations = torch.stack(pred_rotations).cpu().numpy()

    valid_gt_indices = np.linalg.norm(gt_rotations, axis=1) != 0
    valid_pred_indices = np.linalg.norm(pred_rotations, axis=1) != 0

    gt_positions = gt_positions[valid_gt_indices]
    gt_rotations = gt_rotations[valid_gt_indices]

    pred_positions = pred_positions[valid_pred_indices]
    pred_rotations = pred_rotations[valid_pred_indices]

    gt_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in gt_rotations])
    pred_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in pred_rotations])

    # Add rotation values to translation values
    gt_full_poses = gt_positions + gt_euler_angles
    pred_full_poses = pred_positions + pred_euler_angles

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(gt_full_poses[:, 0], gt_full_poses[:, 1], gt_full_poses[:, 2], label="Ground Truth")
    ax.scatter(pred_full_poses[:, 0], pred_full_poses[:, 1], pred_full_poses[:, 2], label="Prediction")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained models
vision_model = VisionOnlyNetwork().to(device)
inertial_model = InertialOnlyNetwork().to(device)
visual_inertial_model = VisualInertialNetwork().to(device)

vision_model.load_state_dict(torch.load("/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/vision_model/vision_model_epoch_50.pth"))
inertial_model.load_state_dict(torch.load("/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/inertial_model/inertial_model_epoch_50.pth"))
visual_inertial_model.load_state_dict(torch.load("/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/visual_inertial_model/visual_inertial_model_epoch_50.pth"))

# Loss functions
# vision_loss_fn = TranslationRotationLoss()
# inertial_loss_fn = TranslationRotationLoss()
# visual_inertial_loss_fn = TranslationRotationLoss()
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
# Evaluate models
vision_loss, vision_gt_positions, vision_pred_positions, vision_gt_rotations, vision_pred_rotations = evaluate_model(vision_model, test_dataloader, device)
inertial_loss, inertial_gt_positions, inertial_pred_positions, inertial_gt_rotations, inertial_pred_rotations = evaluate_model(inertial_model, test_dataloader, device)
visual_inertial_loss, visual_inertial_gt_positions, visual_inertial_pred_positions, visual_inertial_gt_rotations, visual_inertial_pred_rotations = evaluate_model(visual_inertial_model, test_dataloader, device)


print("Vision Only Loss:", vision_loss) 
print("Inertial Only Loss:", inertial_loss)
print("Visual Inertial Loss:", visual_inertial_loss)

# scaling_factor = compute_scaling_factor(vision_gt_positions, vision_pred_positions)
# scaled_pred_positions = [pos * scaling_factor for pos in vision_pred_positions]

# Plot trajectories
plot_trajectory(vision_gt_positions, vision_pred_positions, "Vision Only Network - Translation")
plot_trajectory(inertial_gt_positions, inertial_pred_positions, "Inertial Only Network - Translation")
plot_trajectory(visual_inertial_gt_positions, visual_inertial_pred_positions, "Visual Inertial Network - Translation")

plot_rotations(vision_gt_rotations, vision_pred_rotations, "Vision Only Network - Rotation")
plot_rotations(inertial_gt_rotations, inertial_pred_rotations, "Inertial Only Network - Rotation")
plot_rotations(visual_inertial_gt_rotations, visual_inertial_pred_rotations, "Visual Inertial Network - Rotation")

scatter_trajectory_3d(vision_gt_positions, vision_pred_positions, vision_gt_rotations, vision_pred_rotations, "Vision only 3d Trajectory")

scatter_trajectory_3d(inertial_gt_positions, inertial_pred_positions, inertial_gt_rotations, inertial_pred_rotations, "Inertial only 3d Trajectory")

scatter_trajectory_3d(visual_inertial_gt_positions, visual_inertial_pred_positions, visual_inertial_gt_rotations, visual_inertial_pred_rotations, "Visual Inertial 3d Trajectory")
