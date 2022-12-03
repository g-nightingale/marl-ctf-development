import numpy as np
from collections import deque
from dqn_network import Net
import copy
import random
import torch
import torch.nn as nn
import os

class DQNAgent:
    def __init__(self, 
            q_network=Net(), 
            batch_size=128,
            gamma=0.9,
            lr=0.0005,
            epsilon=0.3,
            target_update_steps=500,
            mem_size=2000,
            n_actions=5,
            device='cpu',
            available_actions=np.array([1, 1, 1, 1, 1]),
            name='dqn',
            chkpt_dir='models/'):

        self.q_network = q_network
        self.target_network = copy.deepcopy(q_network)
        self.target_network.load_state_dict(q_network.state_dict())

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.target_update_steps = target_update_steps
        self.memory = deque(maxlen=mem_size)
        self.n_actions = n_actions
        self.device = device
        # up, down, left, right, nothing
        self.available_actions = available_actions

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def choose_action(self, state):
        """
        Choose an action.
        """

        qval = self.q_network(state)
        qval_ = qval.data.cpu().numpy()
        if (random.random() < self.epsilon):
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(qval_)

        # apply mask based on agents abilities
        #action *= self.available_actions
        return action

    def update_network(self, step_count):
        """
        Update the agent's network.
        """
        minibatch = random.sample(self.memory, self.batch_size)
        state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)
        Q1 = self.q_network(state1_batch) 
        with torch.no_grad():
            Q2 = self.target_network(state2_batch)
        
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step_count % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save_models(self):
        self.save(self.q_network.state_dict(), self.chkpt_file)
    
    def load_models(self):
        self.load_state_dict(torch.load(self.chkpt_file))



    
