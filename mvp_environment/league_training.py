import numpy as np
import torch
from IPython.display import clear_output
import wandb
from gridworld_ctf_mvp import GridworldCtf
from agent_network import Agent
from ppo import PPOTrainer
from metrics_logger import MetricsLogger

class LeagueTrainer:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.opponent_selection_weights = (0.5, 0.3, 0.2)

        self._init_objects()

    def _init_objects(self):
        # Initialise environment
        self.env = GridworldCtf(**self.args.env_config)
        
        dims_data = self.env.get_env_dims()
        local_grid_dims = dims_data[0]
        global_grid_dims = dims_data[1]
        local_metadata_dims = dims_data[2]
        global_metadata_dims = dims_data[3]
        n_channels = local_grid_dims[0]

        self.ppotrainer = PPOTrainer(self.args, local_grid_dims, local_metadata_dims)
        # Setup metrics logger
        self.metlog = MetricsLogger(self.args.n_main_agents)

        #----------------------------------------------------------------------
        # Initialise agent population
        #----------------------------------------------------------------------
        self.main_agents = []
        self.exploiters = []
        self.historical_agents = []

        for _ in range(self.args.n_main_agents):
            self.main_agents.append(Agent(n_channels, self.env.GRID_SIZE, local_metadata_dims[0]).to(self.args.device))

        for _ in range(self.args.n_exploiters):
            self.exploiters.append(Agent(n_channels, self.env.GRID_SIZE, local_metadata_dims[0]).to(self.args.device))

    def duel(self, agent, opponent, max_steps=256, render=False, sleep_time=0.01):
        """
        Duelling algorithm.
        """

        step_count = 0
        done = False
        device = 'cpu'
        self.env.reset()

        while not done:
            step_count += 1

            actions = []
            
            for agent_idx in np.arange(self.env.N_AGENTS):
                # Get global and local states
                if self.env.AGENT_TEAMS[agent_idx]==0:
                    grid_state = torch.tensor(self.env.standardise_state(agent_idx, use_ego_state=True), dtype=torch.float32).to(device)
                    metadata_state = torch.tensor(self.env.get_env_metadata_local(agent_idx), dtype=torch.float32).to(device)
                    action = agent.get_action(grid_state, metadata_state)
                else:
                    grid_state = torch.tensor(self.env.standardise_state(agent_idx, use_ego_state=True, reverse_grid=True), dtype=torch.float32).to(device)
                    metadata_state = torch.tensor(self.env.get_env_metadata_local(agent_idx), dtype=torch.float32).to(device)
                    action = opponent.get_action(grid_state, metadata_state)
                    action = self.env.get_reversed_action(action)

                actions.append(action)

            _, _, done = self.env.step(actions)

            if render:
                self.env.render(sleep_time=sleep_time)
                print(step_count, self.env.metrics['team_flag_captures'][0], self.env.metrics['team_flag_captures'][1])

            if step_count > max_steps:
                done = True

        return self.env.metrics['team_flag_captures'][0], self.env.metrics['team_flag_captures'][1]


    def select_opponent(self, iteration, agent_idx):
        """
        Select agents for training.
        """
        agent_types = ['main', 'exploiter', 'historical']
        weights = self.opponent_selection_weights

        # Adjust weights if we do not have any historical agents yet
        if len(self.historical_agents) == 0:
            weights = (weights[0]/sum(weights[:2]), weights[1]/sum(weights[:2]), 0.0)

        # Choose opponent
        opponent_type = np.random.choice(agent_types, p=weights)
        opponent_idx = np.random.choice(np.arange(len(self.main_agents)) if opponent_type == 'main' else
                                        np.arange(len(self.exploiters)) if opponent_type == 'exploiter' else
                                        np.arange(len(self.historical_agents)))
                                
        if opponent_type == 'main':
            opponent = self.main_agents[opponent_idx] 
        elif opponent_type == 'exploiter':
            opponent = self.exploiters[opponent_idx]
        else:
            opponent = self.historical_agents[opponent_idx]

        # Ensure agent1 and agent2 are different
        while self.main_agents[agent_idx] == opponent:
            opponent_type = np.random.choice(agent_types, p=weights)
            opponent_idx = np.random.choice(np.arange(len(self.main_agents)) if opponent_type == 'main' else
                                            np.arange(len(self.exploiters)) if opponent_type == 'exploiter' else
                                            np.arange(len(self.historical_agents)))
            if opponent_type == 'main':
                opponent = self.main_agents[opponent_idx] 
            elif opponent_type == 'exploiter':
                opponent = self.exploiters[opponent_idx]
            else:
                opponent = self.historical_agents[opponent_idx]

        print(f'Iteration: {iteration} Training main agent {agent_idx} vs {opponent_type} agent {opponent_idx}')
        return opponent

    def update_main_agents(self, win_rate_dict):
        """
        Update agents based on relative performance.
        """
        # Get the probability of adding each main agent, based on their win rates
        win_rates = np.array([v for v in win_rate_dict.values()])
        probs = win_rates/ sum(win_rates)

        if max(win_rates) - min(win_rates) > self.args.max_win_rate_diff:
            # Get the main agent to add
            agent_to_add = np.argmax(probs)
            agent_to_remove = np.argmin(probs)

            print(f'Copying agent {agent_to_add}, removing agent {agent_to_remove}')

            # Add the main agent to the historical agents pool
            new_agent = self.main_agents[agent_to_add]
            self.main_agents.append(new_agent)

            # Maintain the pool size
            del self.main_agents[agent_to_remove]

    def update_exploiters(self):
        """
        Train exploiters against main agents.
        """
        for exploiter_idx, exploiter in enumerate(self.exploiters):
            print(f'Updating exploiter {exploiter_idx}')
            main_agent = np.random.choice(self.main_agents)  # Select a main agent to target

            self.ppotrainer.train_ppo(self.args, self.env, exploiter, main_agent)
            clear_output()

    def update_historical_agents(self, win_rate_dict):
        """
        Add main agents to history.
        """
        # Get the probability of adding each main agent, based on their win rates
        win_rates = np.array([v for v in win_rate_dict.values()])
        probs = win_rates / sum(win_rates)

        # Get the main agent to add
        main_agent_to_add = np.random.choice(self.main_agents, probs, k=1)

        # Add the main agent to the historical agents pool
        self.historical_agents.append(main_agent_to_add)

        # Maintain the pool size
        if len(self.historical_agents) > self.args.n_historical_agents:
            self.historical_agents.pop(0)  # Remove the oldest agent

    def train_league(self):
        
        if self.args.use_wandb_selfplay:
            wandb.init(project=self.args.wandb_project_name,
                        name=self.args.exp_name,
                        config=vars(self.args))
                        
        #----------------------------------------------------------------------
        # League Training Start
        #----------------------------------------------------------------------
        for iteration in range(self.args.number_of_iterations):

            for agent_idx, agent in enumerate(self.main_agents):
                #TODO: Update to select multiple opponents and return as a list
                # Select agent to train and opponent
                opponent = self.select_opponent(iteration, agent_idx)

                #TODO: Update train_ppo method to accept a list of opponents
                # Train PPO
                self.ppotrainer.train_ppo(self.args, self.env, agent, opponent)
                clear_output()

            #----------------------------------------------------------------------
            # Duelling Phase and harvest of metrics
            #----------------------------------------------------------------------
            print(f'Iteration: {iteration} Duelling...')
            all_agents = self.main_agents + self.exploiters + self.historical_agents
            win_rate_dict = {i:0 for i in range(len(self.main_agents))}
            for agent_idx, agent in enumerate(self.main_agents):
                for opponent_idx, opponent in enumerate(all_agents):
                    if agent_idx != opponent_idx:
                        # print(f'agent {agent_idx} vs opponent {opponent_idx}')
                        for _ in range(self.args.number_of_duels):
                            scaling_factor = 1 / (len(all_agents) - 1 + self.args.number_of_duels)
                            agent_score, opponent_score = self.duel(agent, opponent)
                            if agent_score > opponent_score:
                                win_rate_dict[agent_idx] += 1 / (self.args.number_of_duels * len(all_agents)-1)

                            # Log metrics -> track for main agents
                            self.metlog.harvest_metrics(self.env, agent_idx, iteration, scaling_factor)

            print(f'Win rates:\n{win_rate_dict}')

            # Log metrics
            if self.args.use_wandb_selfplay:
                self.metlog.log_to_wandb()

            #----------------------------------------------------------------------
            # Update agent pools
            #----------------------------------------------------------------------
            if iteration % self.args.main_agent_update_interval == 0:
                print(f'Iteration: {iteration} Updating main agents')
                self.update_main_agents(win_rate_dict)

            if iteration % self.args.exploiter_update_interval == 0:
                print(f'Iteration: {iteration} Updating exploiters')
                self.update_exploiters()

            if iteration % self.args.historical_update_interval == 0:
                print(f'Iteration: {iteration} Updating historical agent pool')
                self.update_historical_agents(win_rate_dict)

        # Close wandb session
        if self.args.use_wandb_selfplay:
            self.metlog.log_matplotlib_plots_to_wandb()
            # metlog.log_wandb_table_plots()
            wandb.finish()