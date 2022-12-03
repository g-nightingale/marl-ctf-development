import torch
import torch.nn as nn
import torch.nn.functional as F

# cnn model
# https://stackoverflow.com/questions/51700729/how-to-construct-a-network-with-two-inputs-in-pytorch
# Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
class DQNNetwork(nn.Module):
    def __init__(self, use_device=None):
        super().__init__()
        # channels / filters / filter size
        self.conv1 = nn.Conv2d(1, 3, 3)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 3)

        self.fc_other_info1 = nn.Linear(8, 16)
        self.fc_other_info2 = nn.Linear(16, 8)

        self.fc1 = nn.Linear(8*6*6+8, 128)
        self.fc2 = nn.Linear(128, 5)
        #self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if use_device is not None:
            self.to(use_device)

    def forward(self, state_grid, state_metadata):
        x1 = F.relu(self.conv1(state_grid))
        x1 = F.relu(self.conv2(x1))
        x1 = torch.flatten(x1, 1) # flatten all dimensions except batch

        x2 = F.relu(self.fc_other_info1(state_metadata))
        x2 = F.relu(self.fc_other_info2(x2))

        x = torch.concat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x