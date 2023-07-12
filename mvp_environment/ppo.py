import argparse
import os
import random
import time
from distutils.util import strtobool

# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from IPython.display import clear_output
import wandb
import ray

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOTrainer:
    def __init__(self, args, grid_dims, metadata_dims):
        self.args = args
        self.grid_dims = grid_dims
        self.metadata_dims = metadata_dims
        self.global_step = 0

    def get_single_rollout(self, 
                    env, 
                    agent, 
                    opponent):
        """
        Perform a rollout of the environment.

        Args:
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.

        Returns:
            float: The area of the rectangle.
        """
        
        # Init arrays
        rollout_grid_states = torch.zeros((self.num_steps, ) + self.grid_dims).to(self.device)
        rollout_metadata_states = torch.zeros((self.num_steps, ) + self.metadata_dims).to(self.device)
        rollout_actions = torch.zeros((self.num_steps, )).to(self.device)
        rollout_use_action_mask = torch.zeros((self.num_steps, )).to(self.device)
        rollout_logprobs = torch.zeros((self.num_steps, )).to(self.device)
        rollout_rewards = torch.zeros((self.num_steps, )).to(self.device)
        rollout_dones = torch.zeros((self.num_steps, )).to(self.device)
        rollout_values = torch.zeros((self.num_steps, )).to(self.device)

        # Reset environment
        env.reset()

        for step in range(0, self.num_steps, self.num_agents_per_team):
            # Init actions list to feed into environment
            actions_list = []

            with torch.no_grad():
                
                agent_offset = 0
                for agent_idx in np.arange(env.N_AGENTS):
                    if env.AGENT_TEAMS[agent_idx]==self.team_to_train:
                        use_action_mask = torch.tensor(env.AGENT_TYPE_ACTION_MASK[env.AGENT_TYPES[agent_idx]], dtype=torch.float32).to(self.device)
                        grid_state = torch.tensor(env.standardise_state(agent_idx, reverse_grid=self.reverse_grid), dtype=torch.float32).to(self.device)
                        metadata_state = torch.tensor(env.get_env_metadata(agent_idx), dtype=torch.float32).to(self.device)
                        action, logprob, _, value = agent.get_action_and_value(grid_state, metadata_state, use_action_mask)

                        # Collect values, actions and logprobs for agent (and not opponent)
                        rollout_grid_states[step + agent_offset] = grid_state
                        rollout_metadata_states[step + agent_offset] = metadata_state
                        rollout_values[step + agent_offset] = value.flatten()
                        rollout_actions[step + agent_offset] = action
                        rollout_use_action_mask[step + agent_offset] = use_action_mask
                        rollout_logprobs[step + agent_offset] = logprob
                        if self.reverse_grid:
                            action = env.get_reversed_action(action.item())
                        else:
                            action = action.item()
                        agent_offset += 1
                    else:
                        use_action_mask = torch.tensor(env.AGENT_TYPE_ACTION_MASK[env.AGENT_TYPES[agent_idx]], dtype=torch.float32).to(self.device)
                        grid_state = torch.tensor(env.standardise_state(agent_idx, reverse_grid=not self.reverse_grid), dtype=torch.float32).to(self.device)
                        metadata_state = torch.tensor(env.get_env_metadata(agent_idx), dtype=torch.float32).to(self.device)
                        action, _, _, _ = opponent.get_action_and_value(grid_state, metadata_state, use_action_mask)
                        if not self.reverse_grid:
                            action = env.get_reversed_action(action.item())
                        else:
                            action = action.item()

                    actions_list.append(action)

            # Step the environment
            _, reward, done = env.step(actions_list)
            done_int = 1 if done else 0 

            # TODO: Get global metadata
            #global_metadata_state = torch.tensor(env.get_env_metadata_global(actions_list), dtype=torch.float32).to(self.device)
            
            # Store rewards and global metadata
            agent_offset = 0
            for agent_idx in np.arange(env.N_AGENTS):
                if env.AGENT_TEAMS[agent_idx]==self.team_to_train:
                    rollout_rewards[step + agent_offset] = torch.tensor([reward[agent_idx]]).to(self.device).view(-1)
                    agent_offset += 1
            
            next_done = torch.Tensor([done_int]).to(self.device)

            if rollout_rewards.sum().item() > self.max_rewards:
                self.max_rewards = rollout_rewards.sum().item()

        # Use the first agent in team 1 for the next states
        first_agent_idx = min([k for k, v in env.AGENT_TEAMS.items() if v==self.team_to_train])
        next_grid_state = torch.tensor(env.standardise_state(first_agent_idx, reverse_grid=self.reverse_grid), dtype=torch.float32).to(self.device)
        next_metadata_state = torch.tensor(env.get_env_metadata(first_agent_idx), dtype=torch.float32).to(self.device)

        return rollout_grid_states, \
                rollout_metadata_states, \
                rollout_actions, \
                rollout_use_action_mask, \
                rollout_logprobs, \
                rollout_rewards, \
                rollout_dones, \
                rollout_values, \
                next_grid_state, \
                next_metadata_state, \
                next_done
    
    def calculate_advantages(self, 
                             agent,
                             next_grid_state,
                             next_metadata_state,
                             rewards,
                             next_done,
                             dones,
                             values):
        """
        Calculate advantages.
        """

        with torch.no_grad():
            next_value = agent.get_value(next_grid_state, next_metadata_state).reshape(1, -1)
            if self.args.gae:
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(self.device)
                for t in reversed(range(self.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.args.gamma * nextnonterminal * next_return
                advantages = returns - values

        return advantages, returns
    
    def optimise(self, 
                 agent, 
                 b_grid_states,
                 b_metadata_states,
                 b_logprobs,
                 b_actions,
                 b_use_action_mask,
                 b_advantages,
                 b_returns,
                 b_values
                 ):
        """
        Optimise.
        """
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_grid_states[mb_inds], b_metadata_states[mb_inds], b_use_action_mask[mb_inds], b_actions.long()[mb_inds], )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    break

        return v_loss.item(), pg_loss.item(), entropy_loss.item()
    
    def get_random_opponent(self, opponents):
        """
        Get a random opponent from a list of opponents.
        """
        return np.random.choice(opponents)

    def train_ppo(self, args, env, agent, opponent, train_team1=True, verbose=True):
        run_name = "Run"

        if self.args.use_wandb_ppo:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        @ray.remote
        def ray_rollout(env, agent, opponent):
            return self.get_single_rollout(env, agent, opponent)
    
        # TODO: Move this into trainer script
        # random.seed(self.args.seed)
        # np.random.seed(self.args.seed)
        # torch.manual_seed(self.args.seed)
        # torch.backends.cudnn.deterministic = self.args.torch_deterministic

        if train_team1:
            self.reverse_grid = False
            self.team_to_train = 0
        else:
            self.reverse_grid = True
            self.team_to_train = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.optimizer = optim.Adam(agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Note that because each agent in the team moves once per timestep, we need to multiply the
        # number of steps by the number of agents in the team
        self.num_agents_per_team = env.N_AGENTS // 2
        self.num_steps = self.args.num_steps * self.num_agents_per_team
        self.batch_size = int(self.args.num_envs * self.args.num_steps * self.num_agents_per_team)
        self.minibatch_size = int(self.batch_size // self.args.num_minibatches)
        # num_updates = (self.args.total_timesteps * self.num_agents_per_team) // self.batch_size
        num_updates = (self.args.total_timesteps * self.num_agents_per_team) // int(self.args.num_steps * self.num_agents_per_team)

        print(f'Batch size: {self.batch_size}')
        print(f'Minibatch size: {self.minibatch_size}')

        # Init arrays
        grid_states = torch.zeros((self.num_steps, self.args.num_envs) + self.grid_dims).to(self.device)
        metadata_states = torch.zeros((self.num_steps, self.args.num_envs) + self.metadata_dims).to(self.device)
        actions = torch.zeros((self.num_steps, self.args.num_envs)).to(self.device)
        use_action_mask = torch.zeros((self.num_steps, self.args.num_envs)).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.args.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        start_time = time.time()
        
        self.max_rewards = -np.inf
        total_rewards = []

        #----------------------------------------------------------------------
        # Training Loop Start
        #----------------------------------------------------------------------
        for update in range(1, num_updates + 1):
            self.global_step += (self.args.num_envs * self.args.num_steps)
            
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            #----------------------------------------------------------------------
            # Get rollout
            #----------------------------------------------------------------------
            if self.args.num_envs == 1:
                # opponent = self.get_random_opponent(opponents)

                # Get rollout data
                rollout_data = self.get_single_rollout(env, agent, opponent)
                
                # Load rollout data into global arrays
                grid_states[:, 0, :, :, :] = rollout_data[0]
                metadata_states[:, 0, :] = rollout_data[1]
                actions[:, 0] = rollout_data[2]
                use_action_mask[:, 0] = rollout_data[3]
                logprobs[:, 0] = rollout_data[4]
                rewards[:, 0] = rollout_data[5]
                dones[:, 0] = rollout_data[6]
                values[:, 0] = rollout_data[7]
                next_grid_state = rollout_data[8]
                next_metadata_state = rollout_data[9]
                next_done = rollout_data[10]

            elif self.args.num_envs > 1:
                if self.args.parallel_rollouts:
                    async_results = []

                    # Use apply_async to run the 'get_rollout' method in parallel
                    for _ in range(self.args.num_envs):
                        # opponent = self.get_random_opponent(opponents)
                        async_result = ray_rollout.remote(env, agent, opponent)
                        async_results.append(async_result)

                    # Collect the results
                    rollout_data_multiple = ray.get(async_results)

                    for r_idx, rollout_data in enumerate(rollout_data_multiple):
                        grid_states[:, r_idx, :, :, :] = rollout_data[0]
                        metadata_states[:, r_idx, :] = rollout_data[1]
                        actions[:, r_idx] = rollout_data[2]
                        use_action_mask[:, r_idx] = rollout_data[3]
                        logprobs[:, r_idx] = rollout_data[4]
                        rewards[:, r_idx] = rollout_data[5]
                        dones[:, r_idx] = rollout_data[6]
                        values[:, r_idx] = rollout_data[7]
                        next_grid_state = rollout_data[8]
                        next_metadata_state = rollout_data[9]
                        next_done = rollout_data[10]

                        # Update max rewards
                        if rollout_data[5].sum().item() > self.max_rewards:
                            self.max_rewards = rollout_data[5].sum().item()

                else:
                    for r_idx in range(self.args.num_envs):
                        # Randomly select opponent
                        # opponent = self.get_random_opponent(opponents)
                        
                        # Get rollout data
                        rollout_data = self.get_single_rollout(env, agent, opponent)
                        
                        # Populate arrays
                        grid_states[:, r_idx, :, :, :] = rollout_data[0]
                        metadata_states[:, r_idx, :] = rollout_data[1]
                        actions[:, r_idx] = rollout_data[2]
                        use_action_mask[:, r_idx] = rollout_data[3]
                        logprobs[:, r_idx] = rollout_data[4]
                        rewards[:, r_idx] = rollout_data[5]
                        dones[:, r_idx] = rollout_data[6]
                        values[:, r_idx] = rollout_data[7]
                        next_grid_state = rollout_data[8]
                        next_metadata_state = rollout_data[9]
                        next_done = rollout_data[10]

                        # Update max rewards
                        if rollout_data[5].sum().item() > self.max_rewards:
                            self.max_rewards = rollout_data[5].sum().item()

            #----------------------------------------------------------------------
            # Calculate Advantages
            #----------------------------------------------------------------------
            advantages, returns = self.calculate_advantages(agent,
                                                            next_grid_state,
                                                            next_metadata_state,
                                                            rewards,
                                                            next_done,
                                                            dones,
                                                            values)

            #----------------------------------------------------------------------
            # Update Policy 
            #----------------------------------------------------------------------
            
            # flatten the batch
            b_grid_states = grid_states.reshape((-1,) + self.grid_dims)
            b_metadata_states = metadata_states.reshape((-1,) + self.metadata_dims)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1,)
            b_use_action_mask = use_action_mask.reshape(-1,)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            assert((self.args.num_envs * self.args.num_steps) == (b_grid_states.shape[0] // self.num_agents_per_team))
            
            # Check shapes
            # print('obs:', obs.shape, b_obs.shape)
            # print('obs2:', obs2.shape, b_obs2.shape)
            # print('log probs:', logprobs.shape, b_logprobs.shape)
            # print('actions:', actions.shape, b_actions.shape)
            # print('advantages:', advantages.shape, b_advantages.shape)
            # print('returns:', returns.shape, b_returns.shape)
            # print('values:', values.shape, b_values.shape)
            # print(breakit)

            v_loss, pg_loss, entropy_loss = self.optimise(agent, 
                                                            b_grid_states,
                                                            b_metadata_states,
                                                            b_logprobs,
                                                            b_actions,
                                                            b_use_action_mask,
                                                            b_advantages,
                                                            b_returns,
                                                            b_values)
           
            #----------------------------------------------------------------------
            # Logging
            #----------------------------------------------------------------------
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            total_rewards.append(rewards.sum().item()/self.args.num_envs)
            average_rewards = round(np.mean(total_rewards[-5:]), 2)

            if verbose:
                print(f'i: {update}\t',
                        f'steps: {self.global_step}\t',  
                        f'ar: {average_rewards:.2f}\t',
                        f'mx: {self.max_rewards}\t', 
                        f'lr: {round(self.optimizer.param_groups[0]["lr"], 6)}\t', 
                        f'vl: {round(v_loss, 4)}\t',
                        f'pl: {round(pg_loss, 4)}\t', 
                        f'ent: {round(entropy_loss, 4)}\t')
            # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            # writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print('Complete')
        return total_rewards
