import numpy as np
from collections import deque
import copy
import random
import torch
import torch.nn as nn
import os

class DQNAgent:
    def __init__(self, 
            q_network, 
            batch_size=128,
            gamma=0.9,
            lr=0.0005,
            epsilon=0.3,
            target_update_steps=500,
            mem_size=2000,
            use_softmax=False,
            n_actions=5,
            device='cpu',
            loss='mse',
            available_actions=np.array([1, 1, 1, 1, 1])):

        self.q_network = q_network
        self.target_network = copy.deepcopy(q_network)
        self.target_network.load_state_dict(q_network.state_dict())

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.target_update_steps = target_update_steps
        self.memory = deque(maxlen=mem_size)
        self.use_softmax = use_softmax
        self.n_actions = n_actions
        self.device = device
        # up, down, left, right, nothing
        self.available_actions = available_actions

        if loss == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss == 'huber':
            self.loss_fn = torch.nn.HuberLoss()
        else:
            raise ValueError("Loss function not recognised")

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def softmax_policy(self, qvals, temp=0.9):
        """
        Softmax policy - taken from Deep Reinforcement Learning in Action.
        """
        soft = torch.exp(qvals/temp) / torch.sum(torch.exp(qvals/temp))
        action = torch.multinomial(soft, 1) 
        return action.cpu().numpy().item()

    def choose_action(self, state_grid, state_metadata):
        """
        Choose an action.
        """

        qval = self.q_network(state_grid, state_metadata)
        
        if self.use_softmax:
            action = self.softmax_policy(qval, temp=0.9)
        else:
            if (random.random() < self.epsilon):
                action = np.random.randint(0, self.n_actions)
            else:
                qval_ = qval.data.cpu().numpy()
                action = np.argmax(qval_)

        # apply mask based on agents abilities
        #action *= self.available_actions
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
        
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step_count % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save_model(self):
        self.q_network.save_model()
    
    def load_model(self):
        self.q_network.load_model()



    
