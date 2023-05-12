import torch
import torch.nn as nn
# from data_preprocess_new import custom_collate_train, train_dataset
from data_preprocess import custom_collate, train_dataset, val_dataset
from loss_fn import TranslationRotationLoss, pose_loss
from network import VisionOnlyNetwork, InertialOnlyNetwork, VisualInertialNetwork
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Loss functions
inertial_loss_fn = TranslationRotationLoss()
visual_inertial_loss_fn = TranslationRotationLoss()


# Set your hyperparameters
learning_rate = 1e-4

# Create a network instance
vision_network = VisionOnlyNetwork()
inertial_network = InertialOnlyNetwork()
visual_inertial_network = VisualInertialNetwork()

# Move the network to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_network.to(device)
inertial_network.to(device)
visual_inertial_network.to(device)

# Create an optimizer
vision_optimizer = torch.optim.Adam(vision_network.parameters(), lr=learning_rate)
inertial_optimizer = torch.optim.Adam(inertial_network.parameters(), lr=learning_rate)
visual_inertial_optimizer = torch.optim.Adam(visual_inertial_network.parameters(), lr=learning_rate)

def train_vision_network(model, optimizer, img1, img2, gt_rel_pose, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    optimizer.zero_grad()

    img1 = img1.to(device)
    img2 = img2.to(device)
    gt_rel_pose = gt_rel_pose.to(device)
    
    # Forward pass
    pred_rel_pose = model(img1, img2)
    
    # Compute loss
    # loss = vision_loss_fn(pred_rel_pose, gt_rel_pose)
    loss = pose_loss(pred_rel_pose, gt_rel_pose, alpha=0.5)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()

def train_inertial_network(model, optimizer, imu_data, gt_rel_pose, train=True):
    if train:
        model.train()
    else:
        model.eval()
    optimizer.zero_grad()

    imu_data = imu_data.to(device)
    gt_rel_pose = gt_rel_pose.to(device)

    pred_rel_pose = model(imu_data)
    # loss = inertial_loss_fn(pred_rel_pose, gt_rel_pose)
    loss = pose_loss(pred_rel_pose, gt_rel_pose, alpha=0.5)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()


def train_visual_inertial_network(model, optimizer, img1, img2, imu_seq, gt_rel_pose, train=True):
    if train:
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    img1 = img1.to(device)
    img2 = img2.to(device)
    imu_seq = imu_seq.to(device)
    gt_rel_pose = gt_rel_pose.to(device)

    pred_rel_pose = model(img1, img2, imu_seq)
    # loss = visual_inertial_loss_fn(pred_rel_pose, gt_rel_pose)
    loss = pose_loss(pred_rel_pose, gt_rel_pose, alpha=0.5)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()

# Create lists to store the training loss for each network
vision_losses = []
inertial_losses = []
visual_inertial_losses = []

# Create lists to store the validation loss for each network
vision_val_losses = []
inertial_val_losses = []
visual_inertial_val_losses = []

# Training loop
num_epochs = 50
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Initialize epoch losses
    epoch_vision_loss = 0
    epoch_inertial_loss = 0
    epoch_visual_inertial_loss = 0
    num_batches = 0

    # Training loop
    for i, data in enumerate(train_dataloader):
        if data is None:
            continue

        img1, img2, imu_seq, gt_rel_pose = data

        if img1 is None or img2 is None:
            print("Skipping batch due to missing image(s)")
            continue

        epoch_vision_loss += train_vision_network(vision_network, vision_optimizer, img1, img2, gt_rel_pose)    
        epoch_inertial_loss += train_inertial_network(inertial_network, inertial_optimizer, imu_seq, gt_rel_pose)
        epoch_visual_inertial_loss += train_visual_inertial_network(visual_inertial_network, visual_inertial_optimizer, img1, img2, imu_seq, gt_rel_pose)

        num_batches += 1

    # Calculate average epoch loss and append it to the loss lists
    vision_losses.append(epoch_vision_loss / num_batches)
    inertial_losses.append(epoch_inertial_loss / num_batches)
    visual_inertial_losses.append(epoch_visual_inertial_loss / num_batches)

    # Validation loop
    with torch.no_grad():
        val_vision_loss = 0
        val_inertial_loss = 0
        val_visual_inertial_loss = 0
        num_val_batches = 0

        for i, data in enumerate(val_dataloader):
            if data is None:
                continue

            img1, img2, imu_seq, gt_rel_pose = data

            if img1 is None or img2 is None:
                print("Skipping batch due to missing image(s)")
                continue
            
            val_vision_loss += train_vision_network(vision_network, vision_optimizer, img1, img2, gt_rel_pose, train=False)
            val_inertial_loss += train_inertial_network(inertial_network, inertial_optimizer, imu_seq, gt_rel_pose, train=False)
            val_visual_inertial_loss += train_visual_inertial_network(visual_inertial_network, visual_inertial_optimizer, img1, img2, imu_seq, gt_rel_pose, train=False)

            num_val_batches += 1

        # Calculate average validation loss and append it to the loss lists
        vision_val_losses.append(val_vision_loss / num_val_batches)
        inertial_val_losses.append(val_inertial_loss / num_val_batches)
        visual_inertial_val_losses.append(val_visual_inertial_loss / num_val_batches)

    # Save the model weights after every epoch
    torch.save(vision_network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/vision_model/vision_model_epoch_{}.pth".format(epoch + 1))
    torch.save(inertial_network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/inertial_model/inertial_model_epoch_{}.pth".format(epoch + 1))
    torch.save(visual_inertial_network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/visual_inertial_model/visual_inertial_model_epoch_{}.pth".format(epoch + 1))

# Plot the vision training losses
plt.figure(figsize=(10, 10))
# plt.plot(range(1, num_epochs + 1), inertial_losses, label='Inertial Network')
plt.plot(range(1, num_epochs + 1), vision_losses, label='Visual Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot inertial training loss
plt.figure(figsize=(10, 10))
plt.plot(range(1, num_epochs + 1), inertial_losses, label='Inertial Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot visual inertial training loss
plt.figure(figsize=(10, 10))
# plt.plot(range(1, num_epochs + 1), inertial_losses, label='Inertial Network')
plt.plot(range(1, num_epochs + 1), visual_inertial_losses, label='Visual-Inertial Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot the vision validation loss
plt.figure(figsize=(10, 10))
# plt.plot(range(1, num_epochs + 1), inertial_losses, label='Inertial Network')
plt.plot(range(1, num_epochs + 1), vision_val_losses, label='Visual Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot inertial validation loss
plt.figure(figsize=(10, 10))
plt.plot(range(1, num_epochs + 1), inertial_val_losses, label='Inertial Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot visual inertial validation loss
plt.figure(figsize=(10, 10))
# plt.plot(range(1, num_epochs + 1), inertial_losses, label='Inertial Network')
plt.plot(range(1, num_epochs + 1), visual_inertial_val_losses, label='Visual-Inertial Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()



