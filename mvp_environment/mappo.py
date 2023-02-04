import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size, device='cpu'):
        self.states_grid_local = []
        self.states_grid_global = []
        self.states_metadata_local = []
        self.states_metadata_global = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
        self.device = device

    def generate_batches(self):
        n_states = len(self.states_grid_local)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return T.cat([x for x in self.states_grid_local]).to(self.device), \
                T.cat([x for x in self.states_grid_global]).to(self.device), \
                T.cat([x for x in self.states_metadata_local]).to(self.device), \
                T.cat([x for x in self.states_metadata_global]).to(self.device), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.actions), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches

    def store_memory(self, 
                    state_grid_local, 
                    state_grid_global, 
                    state_metadata_local, 
                    state_metadata_global, 
                    action, 
                    probs, 
                    vals, 
                    reward, 
                    done):
        self.states_grid_local.append(state_grid_local)
        self.states_grid_global.append(state_grid_global)
        self.states_metadata_local.append(state_metadata_local)
        self.states_metadata_global.append(state_metadata_global)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states_grid_local = []
        self.states_grid_global = []
        self.states_metadata_local = []
        self.states_metadata_global = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, 
                grid_len,
                n_actions,
                alpha=0.0003,
                name='ppo_actor',
                n_channels=5,
                conv1_out_channels=32,
                conv1_filter_size=3,
                conv2_out_channels=64,
                conv2_filter_size=2,
                conv3_out_channels=64,
                conv3_filter_size=2,
                fc_other_info1_input_dim=16,
                fc_other_info1_output_dim=64,
                fc1_output_dim=256,
                fc2_output_dim=128,
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

        self.fc1_output_dim = fc1_output_dim
        self.fc2_output_dim = fc2_output_dim

        # Create network shapes
        # in channels / out channels / filter size
        # Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
        self.conv1 = nn.Conv2d(self.n_channels, self.conv1_out_channels, self.conv1_filter_size)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, self.conv2_filter_size)
        self.conv3 = nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, self.conv3_filter_size)

        self.fc_other_info1 = nn.Linear(self.fc_other_info1_input_dim, self.fc_other_info1_output_dim)

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
        self.fc1 = nn.Linear(conv3_unrolled_dim + self.fc_other_info1_output_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc2_output_dim, n_actions)
        self.sm = nn.Softmax(dim=1)
        
        self.device = device
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.n_actions = n_actions

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def softmax_with_temp(self, vals, temp):
        """
        Softmax policy - taken from Deep Reinforcement Learning in Action.
        """
        scaled_qvals = vals/temp
        norm_qvals = scaled_qvals - scaled_qvals.max() 
        soft = T.exp(norm_qvals) / T.sum(T.exp(norm_qvals))
        return soft

    def forward(self, state_grid, state_metadata, temp=1.0):
        x1 = T.relu(self.conv1(state_grid))
        x1 = T.relu(self.conv2(x1))
        x1 = T.relu(self.conv3(x1))
        x1 = T.flatten(x1, 1) # flatten all dimensions except batch

        x2 = T.relu(self.fc_other_info1(state_metadata))

        x3 = T.concat((x1, x2), dim=1)
        x3 = T.relu(self.fc1(x3))
        x3 = T.relu(self.fc2(x3))
        x3 = self.fc3(x3)

        #x3 = self.sm(x3)
        x3 = self.softmax_with_temp(x3, temp=temp)
        dist = Categorical(x3)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class CriticNetwork(nn.Module):
    def __init__(self, 
                grid_len,
                alpha=0.0003,
                name='ppo_critic',
                n_channels=4,
                conv1_out_channels=32,
                conv1_filter_size=3,
                conv2_out_channels=64,
                conv2_filter_size=2,
                conv3_out_channels=64,
                conv3_filter_size=2,
                fc_other_info1_input_dim=16,
                fc_other_info1_output_dim=64,
                fc1_output_dim=256,
                fc2_output_dim=128,
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

        self.fc1_output_dim = fc1_output_dim
        self.fc2_output_dim = fc2_output_dim

        # Create network shapes
        # in channels / out channels / filter size
        # Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
        self.conv1 = nn.Conv2d(self.n_channels, self.conv1_out_channels, self.conv1_filter_size)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, self.conv2_filter_size)
        self.conv3 = nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, self.conv3_filter_size)

        self.fc_other_info1 = nn.Linear(self.fc_other_info1_input_dim, self.fc_other_info1_output_dim)

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
        self.fc1 = nn.Linear(conv3_unrolled_dim + self.fc_other_info1_output_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc2_output_dim, 1)
        
        self.device = device
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state_grid, state_metadata):
        x1 = T.relu(self.conv1(state_grid))
        x1 = T.relu(self.conv2(x1))
        x1 = T.relu(self.conv3(x1))
        x1 = T.flatten(x1, 1) # flatten all dimensions except batch

        x2 = T.relu(self.fc_other_info1(state_metadata))

        x3 = T.concat((x1, x2), dim=1)
        x3 = T.relu(self.fc1(x3))
        x3 = T.relu(self.fc2(x3))
        x3 = self.fc3(x3)

        return x3

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class PPOAgent:
    def __init__(self, 
                 n_actions, 
                 actor_grid_len, 
                 critic_grid_len,
                 actor_metadata_len,
                 critic_metadata_len,
                 gamma=0.99, 
                 alpha=0.0003, 
                 gae_lambda=0.95,
                 policy_clip=0.2, 
                 batch_size=64, 
                 n_epochs=10,
                 softmax_temp=1.0):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.softmax_temp = softmax_temp

        self.actor = ActorNetwork(actor_grid_len, n_actions, alpha, fc_other_info1_input_dim=actor_metadata_len)
        self.critic = CriticNetwork(critic_grid_len, alpha, fc_other_info1_input_dim=critic_metadata_len)
        self.memory = PPOMemory(batch_size)
       
    def store_memory(self, 
                    state_grid_local,
                    state_grid_global, 
                    state_metadata_local,
                    state_metadata_global, 
                    action, 
                    probs, 
                    vals, 
                    reward, 
                    done):
        self.memory.store_memory(state_grid_local,
                                    state_grid_global,
                                    state_metadata_local,
                                    state_metadata_global, 
                                    action,
                                    probs, 
                                    vals, 
                                    reward, 
                                    done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, 
                        state_grid_local,
                        state_metadata_local):
        dist = self.actor(state_grid_local, state_metadata_local, self.softmax_temp)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()

        return action, probs

    def get_state_value(self,
                        state_grid_global, 
                        state_metadata_global):
        value = self.critic(state_grid_global, state_metadata_global)
        value = T.squeeze(value).item()

        return value

    def learn(self):
        for _ in range(self.n_epochs):
            state_grid_arr_local, \
            state_grid_arr_global, \
            state_metadata_arr_local, \
            state_metadata_arr_global, \
            old_prob_arr, \
            vals_arr,\
            action_arr, \
            reward_arr, \
            dones_arr, \
            batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                grid_states_local = state_grid_arr_local[batch]
                grid_states_global = state_grid_arr_global[batch]
                metadata_states_local = state_metadata_arr_local[batch]
                metadata_states_global = state_metadata_arr_global[batch]

                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(grid_states_local, metadata_states_local, self.softmax_temp)
                critic_value = self.critic(grid_states_global, metadata_states_global)

                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()     

        return total_loss          

