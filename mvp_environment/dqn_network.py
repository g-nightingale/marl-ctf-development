import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# cnn model
# https://stackoverflow.com/questions/51700729/how-to-construct-a-network-with-two-inputs-in-pytorch

class DQNNetwork(nn.Module):
    def __init__(self, 
                grid_len,
                name='dqn',
                n_channels=1,
                n_actions=4,
                conv1_out_channels=4,
                conv1_filter_size=3,
                conv2_out_channels=8,
                conv2_filter_size=2,
                conv3_out_channels=12,
                conv3_filter_size=2,
                fc_other_info1_input_dim=16,
                fc_other_info1_output_dim=32,
                fc_other_info2_output_dim=32,
                fc1_output_dim=128,
                chkpt_dir='saved_models',
                device='cpu'):
        super().__init__()

        # Save parameters
        self.grid_len = grid_len
        self.n_channels = n_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv1_filter_size = conv1_filter_size
        self.conv2_out_channels = conv2_out_channels
        self.conv2_filter_size = conv2_filter_size

        self.conv3_out_channels = conv3_out_channels
        self.conv3_filter_size = conv3_filter_size

        self.fc_other_info1_input_dim = fc_other_info1_input_dim
        self.fc_other_info1_output_dim = fc_other_info1_output_dim
        self.fc_other_info2_output_dim = fc_other_info2_output_dim

        self.fc1_output_dim = fc1_output_dim

        # Create network shapes
        # in channels / out channels / filter size
        # Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
        self.conv1 = nn.Conv2d(self.n_channels, self.conv1_out_channels, self.conv1_filter_size)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, self.conv2_filter_size)
        self.conv3 = nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, self.conv3_filter_size)

        self.fc_other_info1 = nn.Linear(self.fc_other_info1_input_dim, self.fc_other_info1_output_dim)
        self.fc_other_info2 = nn.Linear(self.fc_other_info1_output_dim, self.fc_other_info2_output_dim)

        # Calculate number of dimensions for unrolled conv2 layer
        dim1 = self.grid_len - self.conv1_filter_size + 1
        dim2 = dim1 - self.conv2_filter_size + 1
        dim3 = dim2 - self.conv3_filter_size + 1
        conv3_unrolled_dim = self.conv3_out_channels * dim3 * dim3

        print('Network convolutional layer dimensions')
        print(f'Conv 1 output dim: {dim1} x {dim1}')
        print(f'Conv 2 output dim: {dim2} x {dim2}')
        print(f'Conv 3 output dim: {dim3} x {dim3}')
        print(f'Conv 3 unrolled output shape: {conv3_out_channels * dim3 * dim3}\n')

        #self.fc1 = nn.Linear(8*6*6+16, 128)
        self.fc1 = nn.Linear(conv3_unrolled_dim + self.fc_other_info2_output_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, n_actions)
        
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.n_actions = n_actions

    def forward(self, state_grid, state_metadata):
        x1 = F.relu(self.conv1(state_grid))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = torch.flatten(x1, 1) # flatten all dimensions except batch

        x2 = F.relu(self.fc_other_info1(state_metadata))
        x2 = F.relu(self.fc_other_info2(x2))

        x3 = torch.concat((x1, x2), dim=1)
        x3 = F.relu(self.fc1(x3))
        x3 = self.fc2(x3)
        return x3

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt_file)
    
    def load_model(self):
        self.load_state_dict(torch.load(self.chkpt_file))