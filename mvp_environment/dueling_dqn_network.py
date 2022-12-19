import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# cnn model
# https://stackoverflow.com/questions/51700729/how-to-construct-a-network-with-two-inputs-in-pytorch

class DuelingDQNNetwork(nn.Module):
    def __init__(self, 
                name='duelingdqn',
                chkpt_dir='saved_models',
                device='cpu'):
        super().__init__()
        # channels / filters / filter size
        # Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 8, 3)

        self.fc_other_info1 = nn.Linear(12, 16)
        self.fc_other_info2 = nn.Linear(16, 8)

        self.fc1 = nn.Linear(8*6*6+8, 128)
        self.output_value = nn.Linear(128, 1)
        self.output_layer = nn.Linear(128, 4)
        
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)

    def forward(self, state_grid, state_metadata):
        x1 = F.relu(self.conv1(state_grid))
        x1 = F.relu(self.conv2(x1))
        x1 = torch.flatten(x1, 1) # flatten all dimensions except batch

        x2 = F.relu(self.fc_other_info1(state_metadata))
        x2 = F.relu(self.fc_other_info2(x2))

        x = torch.concat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        a = self.output_layer(x)
        v = self.output_value(x).expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt_file)
    
    def load_model(self):
        self.load_state_dict(torch.load(self.chkpt_file))