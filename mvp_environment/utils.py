import numpy as np
import torch


def add_noise(env_dims):
    """
    Add noise to stabilise training.
    """
    return np.random.rand(*env_dims)/100.0

def get_env_metadata(agent_idx, has_flag, device='cpu'):
    """
    Get agent turn and flag holder info.
    """
    agent_turn = np.array([0, 0, 0, 0], dtype=np.int8)
    agent_turn[agent_idx] = 1

    return torch.cat((torch.from_numpy(agent_turn), torch.from_numpy(has_flag))).reshape(1, 8).float().to(device)

def preprocess(grid, max_value=8.0):
    """
    Preprocess the grid to fall between 0 and 1.
    """
    return np.divide(grid, max_value, dtype=np.float16)