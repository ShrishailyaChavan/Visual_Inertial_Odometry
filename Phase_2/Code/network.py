import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

# class VisionOnlyNetwork(nn.Module):
#     def __init__(self):
#         super(VisionOnlyNetwork, self).__init__()
#         self.resnet = resnet50(pretrained=True)
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

#         # Update the input size of the first linear layer to match the concatenated features
#         self.fc = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 7)
#         )

#     def forward(self, img1, img2):
#         img1_features = self.resnet(img1)
#         img2_features = self.resnet(img2)
#         img1_features = img1_features.view(img1_features.size(0), -1)
#         img2_features = img2_features.view(img2_features.size(0), -1)
#         concatenated_features = torch.cat((img1_features, img2_features), dim=1)
#         relative_pose = self.fc(concatenated_features)

#         # Convert the first 3 elements of the output to a quaternion
#         rotation = relative_pose[:, :3]
#         rotation_quaternion = F.normalize(rotation, dim=1)
#         translation = relative_pose[:, 3:]
#         relative_pose_quat = torch.cat((rotation_quaternion, translation), dim=1)

#         return relative_pose_quat

class VisionOnlyNetwork(nn.Module):
    def __init__(self):
        super(VisionOnlyNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(6, 8)),
            nn.Flatten()
        )

        cnn_output_size = 128 * 6 * 8  # Calculate the output size of the last CNN layer (C * H * W)
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, 7)

    def forward(self, img1, img2):
        cnn_img1 = self.cnn(img1)
        cnn_img2 = self.cnn(img2)

        cnn_features = torch.cat((cnn_img1, cnn_img2), dim=1).view(img1.shape[0], 2, -1)
        # cnn_features = cnn_features.unsqueeze(1)  # Add a sequence length dimension

        lstm_out, _ = self.lstm(cnn_features)
        lstm_features = lstm_out[:, -1, :]

        output = self.fc(lstm_features)

        return output

class InertialOnlyNetwork(nn.Module):
    def __init__(self):
        super(InertialOnlyNetwork, self).__init__()

        self.lstm = nn.LSTM(input_size=6, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, imu_data):
        _, (hidden_state, _) = self.lstm(imu_data)
        hidden_state = hidden_state[-1]

        relative_pose = self.fc(hidden_state)

        # Normalize the quaternion part of the output
        normalized_quaternion = F.normalize(relative_pose[:, 3:], dim=1)
        relative_pose = torch.cat((relative_pose[:, :3], normalized_quaternion), dim=1)

        return relative_pose

class VisualInertialNetwork(nn.Module):
    def __init__(self):
        super(VisualInertialNetwork, self).__init__()

        self.vision_network = VisionOnlyNetwork()
        self.inertial_network = InertialOnlyNetwork()

        self.fc = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, img1, img2, imu_data):
        vision_relative_pose = self.vision_network(img1, img2)
        inertial_relative_pose = self.inertial_network(imu_data)

        concatenated_features = torch.cat((vision_relative_pose, inertial_relative_pose), dim=1)
        relative_pose = self.fc(concatenated_features)

        return relative_pose

