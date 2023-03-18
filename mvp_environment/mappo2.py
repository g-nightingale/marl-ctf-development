from argparse import ArgumentParser
import numpy as np
import wandb
from gridworld_ctf_mvp import GridworldCtf
import os
import time
import torch as T
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical
import utils as ut
import multiprocessing as mp

# import pytorch_lightning as pl

CHKPT_DIR = 'saved_models'
NAME = 'ppo'
MODEL_PATH = os.path.join(CHKPT_DIR, NAME)

def parse_args():
    """Pareser program arguments"""
    # Parser
    parser = ArgumentParser()

    # Program arguments (default for Atari games)
    parser.add_argument("--use_wandb", type=bool, help="Log to wandb", default=False)
    parser.add_argument("--max_iterations", type=int, help="Number of iterations of training", default=300)
    parser.add_argument("--n_actors", type=int, help="Number of actors for each update", default=8)
    parser.add_argument("--horizon", type=int, help="Number of timestamps for each actor", default=256)
    parser.add_argument("--epsilon", type=float, help="Epsilon parameter", default=0.2)
    parser.add_argument("--n_epochs", type=int, help="Number of training epochs per iteration", default=3)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=512)
    parser.add_argument("--lr", type=float, help="Learning rate", default=2.5 * 1e-4)
    parser.add_argument("--gamma", type=float, help="Discount factor gamma", default=0.99)
    parser.add_argument("--c1", type=float, help="Weight for the value function in the loss function", default=1)
    parser.add_argument("--c2", type=float, help="Weight for the entropy bonus in the loss function", default=0.01)
    parser.add_argument("--n_test_episodes", type=int, help="Number of episodes to render", default=1)
    parser.add_argument("--seed", type=int, help="Randomizing seed for the experiment", default=0)
    parser.add_argument("--use_mp", type=bool, help="Use parallel processing", default=False)

    # Dictionary with program arguments
    return vars(parser.parse_args())

def get_device():
    """Gets the device (GPU if any) and logs the type"""
    if T.cuda.is_available():
        device = T.device("cuda")
        print(f"Found GPU device: {T.cuda.get_device_name(device)}")
    # elif T.backends.mps.is_available() and T.backends.mps.is_built():
    #      device = T.device("mps")
    #      print(f"Found GPU device: {T.cuda.get_device_name(device)}")
    else:
        device = T.device("cpu")
        print("No GPU found: Running on CPU")
    return device

class ActorNetwork(nn.Module):
    def __init__(self, 
                grid_size,
                n_channels,
                fc_metadata_input_dim,
                n_actions,
                max_iterations,
                n_epochs,
                alpha=0.00003,
                name='ppo_actor',
                conv1_out_channels=32,
                conv1_filter_size=3,
                conv2_out_channels=64,
                conv2_filter_size=2,
                conv3_out_channels=64,
                conv3_filter_size=2,
                fc_metadata_output_dim=64,
                fc1_output_dim=256,
                fc2_output_dim=128,
                chkpt_dir='saved_models',
                device='cpu'):
        super().__init__()

        # Save parameters
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv1_filter_size = conv1_filter_size
        self.conv2_out_channels = conv2_out_channels
        self.conv2_filter_size = conv2_filter_size

        self.conv3_out_channels = conv3_out_channels
        self.conv3_filter_size = conv3_filter_size

        self.fc_metadata_input_dim = fc_metadata_input_dim
        self.fc_metadata_output_dim = fc_metadata_output_dim

        self.fc1_output_dim = fc1_output_dim
        self.fc2_output_dim = fc2_output_dim

        # Create network shapes
        # in channels / out channels / filter size
        # Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
        self.conv1 = nn.Conv2d(self.n_channels, self.conv1_out_channels, self.conv1_filter_size)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, self.conv2_filter_size)
        self.conv3 = nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, self.conv3_filter_size)

        self.fc_metadata = nn.Linear(self.fc_metadata_input_dim, self.fc_metadata_output_dim)

        # Calculate number of dimensions for unrolled conv2 layer
        dim1 = self.grid_size - self.conv1_filter_size + 1
        dim2 = dim1 - self.conv2_filter_size + 1
        dim3 = dim2 - self.conv3_filter_size + 1
        conv3_unrolled_dim = self.conv3_out_channels * dim3 * dim3

        print('Network convolutional layer dimensions')
        print(f'Conv 1 output dim: {dim1} x {dim1}')
        print(f'Conv 2 output dim: {dim2} x {dim2}')
        print(f'Conv 3 output dim: {dim3} x {dim3}')
        print(f'Conv 3 unrolled output shape: {conv3_out_channels * dim3 * dim3}\n')

        #self.fc1 = nn.Linear(8*6*6+16, 128)
        self.fc1 = nn.Linear(conv3_unrolled_dim + self.fc_metadata_output_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc2_output_dim, n_actions)
        self.sm = nn.Softmax(dim=1)
        
        self.device = device
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.n_actions = n_actions

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.scheduler = LinearLR(self.optimizer, 1, 0, max_iterations * n_epochs)

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

        #TODO: Check why x2 doesn't need to be flattened
        x2 = T.relu(self.fc_metadata(state_metadata))

        x3 = T.concat((x1, x2), dim=1)
        x3 = T.relu(self.fc1(x3))
        x3 = T.relu(self.fc2(x3))
        x3 = self.fc3(x3)

        x3 = self.sm(x3)
        #x3 = self.softmax_with_temp(x3, temp=temp)
        dist = Categorical(x3)

        return dist, x3

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class CriticNetwork(nn.Module):
    def __init__(self, 
                grid_size,
                n_channels,
                fc_metadata_input_dim,
                max_iterations,
                n_epochs,
                alpha=0.00003,
                name='ppo_critic',
                conv1_out_channels=32,
                conv1_filter_size=3,
                conv2_out_channels=64,
                conv2_filter_size=2,
                conv3_out_channels=64,
                conv3_filter_size=2,
                fc_metadata_output_dim=64,
                fc1_output_dim=256,
                fc2_output_dim=128,
                chkpt_dir='saved_models',
                device='cpu'):
        super().__init__()

        # Save parameters
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv1_filter_size = conv1_filter_size
        self.conv2_out_channels = conv2_out_channels
        self.conv2_filter_size = conv2_filter_size

        self.conv3_out_channels = conv3_out_channels
        self.conv3_filter_size = conv3_filter_size

        self.fc_metadata_input_dim = fc_metadata_input_dim
        self.fc_metadata_output_dim = fc_metadata_output_dim

        self.fc1_output_dim = fc1_output_dim
        self.fc2_output_dim = fc2_output_dim

        # Create network shapes
        # in channels / out channels / filter size
        # Reminder: height and width of next conv layer = W_1 = [(W_0 + 2P - F)/S] + 1
        self.conv1 = nn.Conv2d(self.n_channels, self.conv1_out_channels, self.conv1_filter_size)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, self.conv2_filter_size)
        self.conv3 = nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, self.conv3_filter_size)

        self.fc_metadata = nn.Linear(self.fc_metadata_input_dim, self.fc_metadata_output_dim)

        # Calculate number of dimensions for unrolled conv2 layer
        dim1 = self.grid_size - self.conv1_filter_size + 1
        dim2 = dim1 - self.conv2_filter_size + 1
        dim3 = dim2 - self.conv3_filter_size + 1
        conv3_unrolled_dim = self.conv3_out_channels * dim3 * dim3

        print('Network convolutional layer dimensions')
        print(f'Conv 1 output dim: {dim1} x {dim1}')
        print(f'Conv 2 output dim: {dim2} x {dim2}')
        print(f'Conv 3 output dim: {dim3} x {dim3}')
        print(f'Conv 3 unrolled output shape: {conv3_out_channels * dim3 * dim3}\n')

        #self.fc1 = nn.Linear(8*6*6+16, 128)
        self.fc1 = nn.Linear(conv3_unrolled_dim + self.fc_metadata_output_dim, self.fc1_output_dim)
        self.fc2 = nn.Linear(self.fc1_output_dim, self.fc2_output_dim)
        self.fc3 = nn.Linear(self.fc2_output_dim, 1)
        
        self.device = device
        if device is not None:
            self.to(device)

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.scheduler = LinearLR(self.optimizer, 1, 0, max_iterations * n_epochs)

    def forward(self, state_grid, state_metadata):
        x1 = T.relu(self.conv1(state_grid))
        x1 = T.relu(self.conv2(x1))
        x1 = T.relu(self.conv3(x1))
        x1 = T.flatten(x1, 1) # flatten all dimensions except batch

        x2 = T.relu(self.fc_metadata(state_metadata))

        # TODO: Check why X2 needs to be flattened here? Why the extra dimension?
        x2 = T.flatten(x2, 1)

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
                 max_iterations,
                 n_epochs,
                 batch_size,
                 alpha,
                 n_actions, 
                 actor_grid_size, 
                 critic_grid_size,
                 actor_channels,
                 critic_channels,
                 actor_metadata_len,
                 critic_metadata_len,
                 c1,
                 c2,
                 device="cpu"):

        # Store attributes
        self.max_iterations = max_iterations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_actions = n_actions 
        self.actor_grid_size = actor_grid_size 
        self.critic_grid_size = critic_grid_size
        self.actor_channels = actor_channels
        self.critic_channels = critic_channels
        self.actor_metadata_len = actor_metadata_len
        self.critic_metadata_len = critic_metadata_len
        self.c1 = c1
        self.c2 = c2
        self.device = device

        # Init annealing schedule and memory buffer
        self.anneals = np.linspace(1, 0, max_iterations)
        self.memory = []
        
        # Build actor and critic networks
        self.actor = ActorNetwork(grid_size=actor_grid_size, 
                                    n_channels=actor_channels,
                                    fc_metadata_input_dim=actor_metadata_len,
                                    n_actions=n_actions, 
                                    max_iterations=max_iterations,
                                    n_epochs=n_epochs,
                                    alpha=alpha)

        self.critic = CriticNetwork(grid_size=critic_grid_size,
                                    n_channels=critic_channels,
                                    fc_metadata_input_dim=critic_metadata_len,
                                    max_iterations=max_iterations,
                                    n_epochs=n_epochs,
                                    #TODO: Temp setting alpha / 2.0
                                    alpha=alpha/2.0)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, grid_state, metadata_state):
        dist, x3 = self.actor(grid_state, metadata_state)
        action = dist.sample()

        # TODO: Check why no need for .item() here
        # probs = T.squeeze(x3) #.item()
        # action = T.squeeze(action) #.item()

        return action, x3 #probs

    def get_state_value(self, grid_state, metadata_state):
        value = self.critic(grid_state, metadata_state)
        # TODO: Check why no need for .item() here
        # value = T.squeeze(value) #.item()

        return value

    def shuffle_memory(self):
        np.random.shuffle(self.memory)

    def clear_memory(self):
        self.memory = []

    def compute_cumulative_rewards(self, gamma):
        """Given a buffer with states, policy action logits, rewards and terminations,
        computes the cumulative rewards for each timestamp and substitutes them into the buffer."""
        curr_rew = 0.

        # Traversing the buffer on the reverse direction
        for i in range(len(self.memory) - 1, -1, -1):
            r, t = self.memory[i][-2], self.memory[i][-1]

            if t:
                curr_rew = 0
            else:
                curr_rew = r + gamma * curr_rew

            self.memory[i][-2] = curr_rew

        # Getting the average reward before normalizing (for logging and checkpointing)
        avg_rew = np.mean([self.memory[i][-2] for i in range(len(self.memory))])

        # Normalizing cumulative rewards
        mean = np.mean([self.memory[i][-2] for i in range(len(self.memory))])
        std = np.std([self.memory[i][-2] for i in range(len(self.memory))]) + 1e-6
        for i in range(len(self.memory)):
            self.memory[i][-2] = (self.memory[i][-2] - mean) / std

        return avg_rew

    def get_losses(self, batch, epsilon, annealing):
        """Returns the three loss terms for a given model and a given batch and additional parameters"""
        # Getting old data
        n = len(batch)
        local_grid_states = T.cat([batch[i][0][0] for i in range(n)])
        local_metadata_states = T.cat([batch[i][0][1] for i in range(n)])

        global_grid_states = T.cat([batch[i][1][0] for i in range(n)])
        global_metadata_states = T.cat([batch[i][1][1] for i in range(n)])

        # TODO: Understand why actions and values are not stored as tensors
        
        actions = T.cat([batch[i][2] for i in range(n)]).view(n, 1)
        #actions = T.tensor([batch[i][2] for i in range(n)])

        logits = T.cat([batch[i][3] for i in range(n)])
        values = T.cat([batch[i][4] for i in range(n)])
        #values = T.tensor([batch[i][4] for i in range(n)])


        cumulative_rewards = T.tensor([batch[i][-2] for i in range(n)]).view(-1, 1).float().to(self.device)

        # Computing predictions with the new model
        _, new_logits = self.choose_action(local_grid_states, local_metadata_states)
        new_values = self.get_state_value(global_grid_states, global_metadata_states)

        # TODO: Temp print
        # print(f'new logits shape: {new_logits.shape}')
        # print(f'logits shape: {logits.shape}')
        # print(f'actions shape: {actions.shape}')
        # print(f'new logits gathers shape: {new_logits.gather(1, actions).shape}')
        # print(f'logits gather shape: {logits.gather(1, actions).shape}')

        # Loss on the state-action-function / actor (L_CLIP)
        advantages = cumulative_rewards - values
        margin = epsilon * annealing
        ratios = new_logits.gather(1, actions) / logits.gather(1, actions)

        l_clip = T.mean(
            T.min(
                T.cat(
                    (ratios * advantages,
                    T.clip(ratios, 1 - margin, 1 + margin) * advantages),
                    dim=1),
                dim=1
            ).values
        )

        # Loss on the value-function / critic (L_VF)
        l_vf = T.mean((cumulative_rewards - new_values) ** 2)

        # Bonus for entropy of the actor
        entropy_bonus = T.mean(T.sum(-new_logits * (T.log(new_logits + 1e-5)), dim=1))

        return l_clip, l_vf, entropy_bonus

    def learn(self, iteration, epsilon):
        
        # Shuffle memory bugger
        self.shuffle_memory()

        # Running optimization for a few epochs
        for _ in range(self.n_epochs):
            for batch_idx in range(len(self.memory) // self.batch_size):
             
                # Getting batch for this buffer
                start = self.batch_size * batch_idx
                end = start + self.batch_size if start + self.batch_size < len(self.memory) else -1
                batch = self.memory[start:end]

                # Zero-ing optimizers gradients
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # Getting the losses
                annealing = self.anneals[iteration]
                l_clip, l_vf, entropy_bonus = self.get_losses(batch, epsilon, annealing)

                # Computing total loss and back-propagating it
                loss = l_clip - self.c1 * l_vf + self.c2 * entropy_bonus
                loss.backward()

                # nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                # nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

                # Optimizing
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
            self.actor.scheduler.step()
            self.critic.scheduler.step()

        # Clear memory buffer
        self.clear_memory()

        return loss, l_clip, l_vf, entropy_bonus


@T.no_grad()
def run_timestamps(env, 
                   agent_t1, 
                   agent_t2,
                   timestamps=128, 
                   render=False, 
                   device="cpu"):
    """Runs the given policy on the given environment for the given amount of timestamps.
     Returns a buffer with state action transitions and rewards."""
    buffer_t1 = []
    buffer_t2 = []
    env.reset()
    
    total_reward = 0
    # Running timestamps and collecting state, actions, rewards and terminations
    for ts in range(timestamps):
        # Collect actions for each agent
        local_states = []
        global_states = []
        actions = []
        action_logits = []
        vals = []
        for agent_idx in np.arange(env.N_AGENTS):
            # Get global and local states
            metadata_state_local_ = env.get_env_metadata_local(agent_idx) 
            metadata_state_local = T.from_numpy(metadata_state_local_).float().to(device)
            
            # Get global and local states
            grid_state_local_ = env.standardise_state(agent_idx, use_ego_state=True)
            grid_state_local = T.from_numpy(grid_state_local_).float().to(device)

            if env.AGENT_TEAMS[agent_idx]==0:
                action, act_logits = agent_t1.choose_action(grid_state_local, metadata_state_local)
            else:
                action, act_logits = agent_t2.choose_action(grid_state_local, metadata_state_local)
            
            # Append actions and probs
            local_states.append((grid_state_local, metadata_state_local))
            actions.append(action)
            action_logits.append(act_logits)

        # Step the environment
        _, reward, done = env.step(actions)

        # Get global metadata state
        metadata_state_global_ = env.get_env_metadata_global(actions)
        metadata_state_global = T.from_numpy(metadata_state_global_).float().to(device)

        for agent_idx in np.arange(env.N_AGENTS):
            # Create the global metadata state: state + actions
            grid_state_global_ = env.standardise_state(agent_idx, use_ego_state=False)
            grid_state_global = T.from_numpy(grid_state_global_).float().to(device)

            if env.AGENT_TEAMS[agent_idx]==0:
                val = agent_t1.get_state_value(grid_state_global, metadata_state_global)
            else:
                val = agent_t2.get_state_value(grid_state_global, metadata_state_global)
                
            global_states.append((grid_state_global, metadata_state_global))
            vals.append(val)

        # Rendering / storing (s, a, r, t) in the buffer
        if render:
            total_reward += reward[0]
            env.render()
            print(f'step: {ts} \treward: {reward} \ttotal reward:{total_reward} \n')
            time.sleep(0.3)
        else:
            # Put the agent rollout data into the correct team buffer
            for agent_idx in np.arange(env.N_AGENTS):
                if env.AGENT_TEAMS[agent_idx]==0:
                    buffer_t1.append([local_states[agent_idx], 
                                    global_states[agent_idx], 
                                    actions[agent_idx], 
                                    action_logits[agent_idx], 
                                    vals[agent_idx], 
                                    reward[agent_idx], 
                                    done])
                else:
                    buffer_t2.append([local_states[agent_idx], 
                                    global_states[agent_idx], 
                                    actions[agent_idx], 
                                    action_logits[agent_idx], 
                                    vals[agent_idx], 
                                    reward[agent_idx], 
                                    done])

        # Resetting environment if episode terminated or truncated
        if done:
            env.reset()


    return buffer_t1, buffer_t2

# Put into PPO agent class
def training_loop(env, 
                    agent_t1, 
                    agent_t2, 
                    max_iterations, 
                    n_actors, 
                    horizon, 
                    gamma, 
                    epsilon, 
                    device, 
                    use_mp=False,
                    use_wandb=False):
    """Train the model on the given environment using multiple actors acting up to n timestamps."""

    # Training variables
    max_reward = float("-inf")

    # global results
    def collect_result(result):
        results.append(result)

    # Training loop
    for iteration in range(max_iterations):
        # Collecting timestamps for all actors with the current policy
        if use_mp:
            results = []
            pool = mp.Pool(mp.cpu_count())
            for _ in range(n_actors):
                pool.apply_async(run_timestamps, 
                                args=(env, agent_t1, agent_t2, horizon, False, device),
                                callback=collect_result)   
            pool.close()
            pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

            # for i in range(n_actors):
            #     buffer.extend(results[i])
        else:
            for _ in range(n_actors):
                new_buffer_data = run_timestamps(env, agent_t1, agent_t2, horizon, False, device)
                agent_t1.memory.extend(new_buffer_data[0])
                agent_t2.memory.extend(new_buffer_data[1])

        # Computing cumulative rewards and shuffling the buffer
        avg_rew_t1 = agent_t1.compute_cumulative_rewards(gamma)
        avg_rew_t2 = agent_t2.compute_cumulative_rewards(gamma)

        # Learn over epochs
        loss_t1, l_clip_t1, l_vf_t1, entropy_bonus_t1 = agent_t1.learn(iteration, epsilon)
        loss_t2, l_clip_t2, l_vf_t2, entropy_bonus_t2 = agent_t2.learn(iteration, epsilon)


        log = f"Iteration {iteration + 1} / {max_iterations}: " \
              f"Average Reward team 1: {avg_rew_t1:.2f}\t" \
              f"Average Reward team 2: {avg_rew_t2:.2f}\t" \
              f"Loss team 1: {loss_t1.item():.3f} " \
              f"Loss team 2: {loss_t2.item():.3f} " \
              f"(L_CLIP_t1: {l_clip_t1.item():.1f} | L_VF_t1: {l_vf_t1.item():.1f} | L_bonus_t1: {entropy_bonus_t1.item():.1f})" \
              f"(L_CLIP_t2: {l_clip_t2.item():.1f} | L_VF_t1: {l_vf_t2.item():.1f} | L_bonus_t2: {entropy_bonus_t2.item():.1f})"
        if avg_rew_t1 > max_reward:
            # T.save(model.state_dict(), MODEL_PATH)
            max_reward = avg_rew_t1
            log += " --> Stored model with highest average reward"
        print(log)

        # Logging information to W&B
        if use_wandb:
            wandb.log({
                "average reward team 1": avg_rew_t1,
                "average reward team 2": avg_rew_t2,
                "loss_t1 (total)": loss_t1.item(),
                "loss_t1 (clip)": l_clip_t1.item(),
                "loss_t1 (vf)": l_vf_t1.item(),
                "loss_t1 (entropy bonus)": entropy_bonus_t1.item(),
                "loss_t2 (total)": loss_t2.item(),
                "loss_t2 (clip)": l_clip_t2.item(),
                "loss_t2 (vf)": l_vf_t2.item(),
                "loss_t2 (entropy bonus)": entropy_bonus_t2.item()
            })

    # Finishing W&B session
    if use_wandb:
        wandb.finish()

def testing_loop(env, agent_t1, agent_t2, n_episodes, device):
    """Runs the learned policy on the environment for n episodes"""
    for _ in range(n_episodes):
        run_timestamps(env, agent_t1, agent_t2, timestamps=128, render=True, device=device)

def main():
    # Parsing program arguments
    args = parse_args()
    print(args)

    # Setting seed
    # pl.seed_everything(args["seed"])

    # Getting device
    device = get_device()

    use_wandb = args['use_wandb']

    # Creating environment (discrete action space)
    env_name = "Multi-Agent-GW-CTF"
    config = {
                'GAME_MODE':'static',
                'GRID_SIZE':6,
                'AGENT_CONFIG':{
                    0: {'team':0, 'type':0},
                    1: {'team':1, 'type':0}
                },
                'DROP_FLAG_WHEN_NO_HP':False
            }


    env = GridworldCtf(**config)

    # Get env dimensions -> need to know these for the actor and critic networks
    #   Grid: (Batch, Channels, Height, Width)
    #   Metadata: (Batch, Length)
    local_grid_dims, global_grid_dims, local_metadata_dims, global_metadata_dims = env.get_env_dims()
    # print(local_grid_dims)
    # print(global_grid_dims)
    # print(local_metadata_dims)
    # print(global_metadata_dims)

    # Creating the model (both actor and critic)
    # model = MyPPO(actor_dims[1], env.action_space).to(device)
    agent_t1 = PPOAgent(max_iterations=args["max_iterations"],
                        n_epochs=args["n_epochs"],
                        batch_size=args["batch_size"],
                        alpha=args["lr"],
                        n_actions=env.ACTION_SPACE,
                        actor_grid_size=local_grid_dims[2], 
                        critic_grid_size=global_grid_dims[2], 
                        actor_channels=local_grid_dims[1],
                        critic_channels=global_grid_dims[1],
                        actor_metadata_len=local_metadata_dims[1], 
                        critic_metadata_len=global_metadata_dims[1],
                        c1=args["c1"],
                        c2=args["c1"])
        
    agent_t2 = PPOAgent(max_iterations=args["max_iterations"],
                        n_epochs=args["n_epochs"],
                        batch_size=args["batch_size"],
                        alpha=args["lr"],
                        n_actions=env.ACTION_SPACE,
                        actor_grid_size=local_grid_dims[2], 
                        critic_grid_size=global_grid_dims[2], 
                        actor_channels=local_grid_dims[1],
                        critic_channels=global_grid_dims[1],
                        actor_metadata_len=local_metadata_dims[1], 
                        critic_metadata_len=global_metadata_dims[1],
                        c1=args["c1"],
                        c2=args["c1"])       
        
    # Starting a new Weights & Biases run
    if use_wandb:
        wandb.init(project="MARL-CTF-GW",
                name=f"PPO - {env_name}",
                config={
                    "env": str(env),
                    "grid_size": env.GRID_SIZE,
                    "n_agents": env.N_AGENTS,
                    "number of actors": args["n_actors"],
                    "horizon": args["horizon"],
                    "gamma": args["gamma"],
                    "epsilon": args["epsilon"],
                    "epochs": args["n_epochs"],
                    "batch size": args["batch_size"],
                    "learning rate": args["lr"],
                    "c1": args["c1"],
                    "c2": args["c2"]
                })

    # Training
    training_loop(env, 
                    agent_t1, 
                    agent_t2,
                    args["max_iterations"], 
                    args["n_actors"], 
                    args["horizon"], 
                    args["gamma"], 
                    args["epsilon"],
                    device, 
                    args["use_mp"],
                    use_wandb)

    # Loading best model
    # model = MyPPO(actor_dims[1], env.action_space).to(device)
    # model.load_state_dict(T.load(MODEL_PATH, map_location=device))

    # # Testing
    # env.reset()
    # testing_loop(env, model, args["n_test_episodes"], device)
    # env.close()

if __name__ == '__main__':
    main()