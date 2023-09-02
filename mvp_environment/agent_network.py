import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
    
class Agent(nn.Module):
    def __init__(self, n_actions, n_channels, grid_size, metadata_size, device='cpu'):
        super(Agent, self).__init__()
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.metadata_size = metadata_size
        self.unrolled_conv_size = self.get_unrolled_conv_size()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.unrolled_conv_size + self.metadata_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        self.device = device

        # Actions masks
        self.mask_5 = torch.tensor([1] * 5 + [0] * (n_actions - 5)).to(self.device)

    def get_unrolled_conv_size(self):
        # Calculate number of dimensions for unrolled conv2 layer
        dim1 = self.grid_size - 3 + 1
        dim2 = dim1 - 3 + 1
        return 32 * dim2 * dim2
    
    def forward(self, x, x2):
        x = torch.tanh(self.conv1(x))        
        x = torch.tanh(self.conv2(x))        
        x = x.view(-1, self.unrolled_conv_size)

        # Add in metadata
        x = torch.concat((x, x2), dim=1)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))   
        return self.value_head(x), self.action_head(x)

    def get_action(self, x, x2, masking_decision_tensor):       
        _, logits = self(x, x2)

        mask_5 = self.mask_5
        if len(logits.shape) == 2:
            mask_5 = self.mask_5.unsqueeze(0)  # Shape becomes [1, 8]
            masking_decision_tensor = masking_decision_tensor.view(-1, 1)  # Shape becomes [batch_size, 1]

        # Use mask_4 if decision is 1, else we'll later replace it with a mask of ones inside the function
        mask = torch.where(masking_decision_tensor == 1, mask_5, torch.zeros_like(mask_5))
        mask = torch.where(masking_decision_tensor == 0, torch.ones_like(mask), mask)

        logits = logits + (mask.float() - 1) * 1e9

        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item()
    
    def get_value(self, x, x2):
        return self(x, x2)[0]

    def get_action_and_value(self, x, x2, masking_decision_tensor, action=None):
        value, logits = self(x, x2)

        mask_5 = self.mask_5
        if len(logits.shape) == 2:
            mask_5 = self.mask_5.unsqueeze(0)  # Shape becomes [1, 8]
            masking_decision_tensor = masking_decision_tensor.view(-1, 1)  # Shape becomes [batch_size, 1]

        # Use mask_4 if decision is 1, else we'll later replace it with a mask of ones inside the function
        mask = torch.where(masking_decision_tensor == 1, mask_5, torch.zeros_like(mask_5))
        mask = torch.where(masking_decision_tensor == 0, torch.ones_like(mask), mask)

        logits = logits + (mask.float() - 1) * 1e9

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
    
class AgentNonMasking(nn.Module):
    def __init__(self, n_actions, n_channels, grid_size, metadata_size, device='cpu'):
        super(AgentNonMasking, self).__init__()
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.metadata_size = metadata_size
        self.unrolled_conv_size = self.get_unrolled_conv_size()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.unrolled_conv_size + self.metadata_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        self.device = device

    def get_unrolled_conv_size(self):
        # Calculate number of dimensions for unrolled conv2 layer
        dim1 = self.grid_size - 3 + 1
        dim2 = dim1 - 3 + 1
        return 32 * dim2 * dim2
    
    def forward(self, x, x2):
        x = torch.tanh(self.conv1(x))        
        x = torch.tanh(self.conv2(x))        
        x = x.view(-1, self.unrolled_conv_size)

        # Add in metadata
        x = torch.concat((x, x2), dim=1)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))   
        return self.value_head(x), self.action_head(x)

    def get_action(self, x, x2, masking_decision_tensor):       
        _, logits = self(x, x2)

        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item()
    
    def get_value(self, x, x2):
        return self(x, x2)[0]

    def get_action_and_value(self, x, x2, masking_decision_tensor, action=None):
        value, logits = self(x, x2)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value