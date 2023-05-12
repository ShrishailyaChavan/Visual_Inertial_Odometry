import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_loss(predicted_pose, gt_pose, alpha=0.5):
    mse_loss = nn.MSELoss()
    position_loss = mse_loss(predicted_pose[:, :3], gt_pose[:, :3])
    orientation_loss = mse_loss(predicted_pose[:, 3:], gt_pose[:, 3:])

    cosine_similarity_loss = nn.CosineSimilarity(dim=1)
    position_similarity = cosine_similarity_loss(predicted_pose[:, :3], gt_pose[:, :3])
    orientation_similarity = cosine_similarity_loss(predicted_pose[:, 3:], gt_pose[:, 3:])

    position_cosine_loss = 1 - position_similarity.mean()
    orientation_cosine_loss = 1 - orientation_similarity.mean()

    combined_position_loss = alpha * position_loss + (1 - alpha) * position_cosine_loss
    combined_orientation_loss = alpha * orientation_loss + (1 - alpha) * orientation_cosine_loss

    return combined_position_loss + combined_orientation_loss

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, prediction, target):
        batch_size = prediction.shape[0]

        # print("prediction shape:", prediction.shape)  # Add this line
        # print("target shape:", target.shape)  # Add this line
        
        prediction_rotations = R.from_quat(prediction.view(batch_size, 4).detach().cpu().numpy())
        target_rotations = R.from_quat(target.view(batch_size, 4).detach().cpu().numpy())

        geodesic_distances = prediction_rotations.inv() * target_rotations
        geodesic_angles = geodesic_distances.magnitude()
        
        return torch.mean(torch.tensor(geodesic_angles ** 2, device=prediction.device))

class TranslationRotationLoss(nn.Module):
    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super(TranslationRotationLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = GeodesicLoss()

    def forward(self, prediction, target):
        translation_prediction = prediction[:, :3]
        translation_target = target[:, :3]
        rotation_prediction = prediction[:, 3:]
        rotation_target = target[:, 3:]

        translation_loss = self.translation_loss(translation_prediction, translation_target)
        rotation_loss = self.rotation_loss(rotation_prediction, rotation_target)

        return self.translation_weight * translation_loss + self.rotation_weight * rotation_loss

