import numpy as np
from collections import deque
import copy
import random
import torch
import torch.nn as nn
import os
from dqn_network import DQNNetwork

class DQNAgent:
    def __init__(self, 
            name, 
            grid_len,
            n_actions=8,
            n_channels=1,
            batch_size=32,
            gamma=0.9,
            lr=0.000025,
            epsilon=1.0,
            temp=0.9,
            target_update_steps=1000,
            mem_size=40000,
            use_softmax=False,
            device='cpu',
            loss='mse',
            ddqn=False,
            soft_update=False,
            tau=0.1):

        self.n_actions = n_actions
        self.n_channels = n_channels
        self.q_network = DQNNetwork(grid_len=grid_len, name=name, device=device, n_actions=self.n_actions, n_channels=self.n_channels)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.temp = temp
        self.target_update_steps = target_update_steps
        self.memory = deque(maxlen=mem_size)
        self.use_softmax = use_softmax
        self.device = device

        if loss == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss == 'huber':
            self.loss_fn = torch.nn.HuberLoss()
        else:
            raise ValueError("Loss function not recognised")

        self.ddqn = ddqn
        self.soft_update = soft_update
        self.tau = tau
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def softmax_policy(self, qvals):
        """
        Softmax policy - taken from Deep Reinforcement Learning in Action.
        """
        scaled_qvals = qvals/self.temp
        norm_qvals = scaled_qvals - scaled_qvals.max() 
        soft = torch.exp(norm_qvals) / torch.sum(torch.exp(norm_qvals))
        action = torch.multinomial(soft, 1) 
        return action.cpu().numpy().item()

    def choose_action(self, state_grid, state_metadata):
        """
        Choose an action.
        """

        qval = self.q_network(state_grid, state_metadata)
        
        if self.use_softmax:
            action = self.softmax_policy(qval)
        else:
            if (random.random() < self.epsilon):
                action = np.random.randint(0, self.n_actions)
            else:
                qval_ = qval.data.cpu().numpy()
                action = np.argmax(qval_)

        return action

    def update_network(self, step_count):
        """
        Update the agent's network.
        """
        minibatch = random.sample(self.memory, self.batch_size)
        state1_grid_batch = torch.cat([x[0][0] for x in minibatch]).to(self.device)
        state1_metadata_batch = torch.cat([x[0][1] for x in minibatch]).to(self.device)
        action_batch = torch.Tensor([x[1] for x in minibatch]).to(self.device)
        reward_batch = torch.Tensor([x[2] for x in minibatch]).to(self.device)
        state2_grid_batch = torch.cat([x[3][0] for x in minibatch]).to(self.device)
        state2_metadata_batch = torch.cat([x[3][1] for x in minibatch]).to(self.device)
        done_batch = torch.Tensor([x[4] for x in minibatch]).to(self.device)

        Q1 = self.q_network(state1_grid_batch, state1_metadata_batch) 
        with torch.no_grad():
            Q2 = self.target_network(state2_grid_batch, state2_metadata_batch)

        if self.ddqn:
            # Use online network to select best action index
            q1_argmax = torch.argmax(Q1, dim=1)
            Y = reward_batch + self.gamma * ((1-done_batch) *  Q2[torch.arange(self.batch_size), q1_argmax])
        else:
            Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2, dim=1)[0])

        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update weights
        if self.soft_update:
            self.soft_weight_update()
        else:
            if step_count % self.target_update_steps == 0:
                self.hard_weight_update()

        return loss.item()

    def hard_weight_update(self):
        """ 
        Apply a hard weight update.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_weight_update(self, tau=None):
        """
        Apply a soft weight update.
        """
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_network.parameters(), 
                                  self.q_network.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def save_model(self):
        self.q_network.save_model()
    
    def load_model(self):
        self.q_network.load_model()



    
