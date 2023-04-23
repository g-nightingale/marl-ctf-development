import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Agent(nn.Module):
    def __init__(self, n_channels, grid_size, metadata_size):
        super(Agent, self).__init__()
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.metadata_size = metadata_size
        self.unrolled_conv_size = self.get_unrolled_conv_size()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.unrolled_conv_size + self.metadata_size, 256)
        self.action_head = nn.Linear(256, 4)
        self.value_head = nn.Linear(256, 1)

    def get_unrolled_conv_size(self):
        # Calculate number of dimensions for unrolled conv2 layer
        dim1 = self.grid_size - 3 + 1
        dim2 = dim1 - 3 + 1
        return 32 * dim2 * dim2

    def get_value(self, x, x2):
        x = torch.tanh(self.conv1(x))        
        x = torch.tanh(self.conv2(x))        
        x = x.view(-1, self.unrolled_conv_size)

        # Add in metadata
        x = torch.concat((x, x2), dim=1)

        x = torch.tanh(self.fc1(x))        
        return self.value_head(x)

    def get_action(self, x, x2, action=None):
        x = torch.tanh(self.conv1(x))        
        x = torch.tanh(self.conv2(x))        
        x = x.view(-1, self.unrolled_conv_size)

        # Add in metadata
        x = torch.concat((x, x2), dim=1)

        x = torch.tanh(self.fc1(x))        
        logits = self.action_head(x)

        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item()

    def get_action_and_value(self, x, x2, action=None):
        x = torch.tanh(self.conv1(x))        
        x = torch.tanh(self.conv2(x))        
        x = x.view(-1, self.unrolled_conv_size)

        # Add in metadata
        x = torch.concat((x, x2), dim=1)

        x = torch.tanh(self.fc1(x))        
        logits = self.action_head(x)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(x)