import numpy as np
from collections import defaultdict
from IPython.display import clear_output
import wandb
from gridworld_ctf import GridworldCtf
from agent_network import Agent
from ppo import PPOTrainer
from metrics_logger import MetricsLogger
import time
import ray
import copy
import os
import pickle
from utils import duel

@ray.remote
def ray_duel(env, agent, opponent, idxs, return_result, device):
    return duel(env, agent, opponent, idxs, return_result, device)

class TimeCapsule:
    def __init__(self,
                 args,
                 symmetric_teams,
                 agent_types,
                 agent_teams,
                 all_agents_t1,
                 all_agents_t2,
                 metlog,
                 winrate_matrices):
        self.args = args
        self.symmetric_teams = symmetric_teams
        self.agent_types = agent_types
        self.agent_teams = agent_teams
        self.all_agents_t1 = all_agents_t1
        self.all_agents_t2 = all_agents_t2
        self.metlog = metlog
        self.winrate_matrices = winrate_matrices

class LeagueTrainer:
    def __init__(self, args):
        self.args = args
        # self.rng = np.random.default_rng(args.seed)
        self.rng = np.random.default_rng()
        self.opponent_selection_weights = {
             'main': (0.3, 0.3, 0.2, 0.2),
             'coach': (1.0, 0.0, 0.0, 0.0),
             'league': (0.34, 0.0, 0.33, 0.33)
        }
        self.winrate_matrices = []
        self._init_objects()

        CWD = os.getcwd()
        RESULTS_FOLDER_NAME = 'runs'
        self.OUTPUT_FOLDER_NAME = CWD + '/' + RESULTS_FOLDER_NAME + '/' + args.exp_name

        self.TIME_CAPSULE_FILE_NAME = 'time_capsule'
        self.FILE_EXT = '.bin'

    def _init_objects(self):
        #----------------------------------------------------------------------
        # Initialise environment and get dimensions
        #----------------------------------------------------------------------
        self.env = GridworldCtf(**self.args.env_config)
        
        dims_data = self.env.get_env_dims()
        self.local_grid_dims = dims_data[0]
        global_grid_dims = dims_data[1]
        self.local_metadata_dims = dims_data[2]
        global_metadata_dims = dims_data[3]
        self.n_channels = self.local_grid_dims[0]

        # Check if teams are symmetrical
        self.symmetric_teams = sorted([v for k, v in self.env.AGENT_TYPES.items() if self.env.AGENT_TEAMS[k] == 0]) == sorted([v for k, v in self.env.AGENT_TYPES.items() if self.env.AGENT_TEAMS[k] == 1])
        if self.args.force_two_teams:
            self.symmetric_teams = False

        if self.args.n_coaching_agents == 0 and self.args.n_league_agents == 0:
            self.opponent_selection_weights['main'] = (1.0, 0.0, 0.0, 0.0)

        # Create agent team types
        self.agent_team_types = defaultdict(list)
        for k, v in self.env.AGENT_TYPES.items():
            self.agent_team_types[self.env.AGENT_TEAMS[k]].append(v)

        #----------------------------------------------------------------------
        # Instantiate ppo trainer and metrics logger
        #----------------------------------------------------------------------
        self.ppotrainer = PPOTrainer(self.args, 
                                    self.local_grid_dims, 
                                    self.local_metadata_dims)
        
        self.metlog = MetricsLogger(self.args.n_main_agents,
                                    self.args.n_coaching_agents,
                                    self.args.n_league_agents,
                                    self.agent_team_types,
                                    [k for k in self.env.AGENT_TYPES.keys()],
                                    self.symmetric_teams)

        self.learning_rewards = defaultdict(lambda: defaultdict(list))

        #----------------------------------------------------------------------
        # Initialise agent population
        #----------------------------------------------------------------------
        self.main_agents_t1 = []
        self.coaching_agents_t1 = []
        self.league_agents_t1 = []
        self.historical_agents_t1 = []
        self.main_agents_t1_prev = []
        self.coaching_agents_t1_prev = []
        self.league_agents_t1_prev = []

        for _ in range(self.args.n_main_agents):
            self.main_agents_t1.append(Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0], self.args.device))
            
        for _ in range(self.args.n_coaching_agents):
            self.coaching_agents_t1.append(Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0], self.args.device))

        for _ in range(self.args.n_league_agents):
            self.league_agents_t1.append(Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0], self.args.device))

        self.main_agents_t1_counter = [0] * self.args.n_main_agents
        self.coaching_agents_t1_counter = [0] * self.args.n_coaching_agents
        self.league_agents_t1_counter = [0] * self.args.n_league_agents

        # Initialise agents if teams are asymmetric
        if not self.symmetric_teams:
            self.main_agents_t2 = []
            self.coaching_agents_t2 = []
            self.league_agents_t2 = []
            self.historical_agents_t2 = []
            self.main_agents_t2_prev = []
            self.coaching_agents_t2_prev = []
            self.league_agents_t2_prev = []

            for _ in range(self.args.n_main_agents):
                self.main_agents_t2.append(Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device))
                
            for _ in range(self.args.n_coaching_agents):
                self.coaching_agents_t2.append(Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device))

            for _ in range(self.args.n_league_agents):
                self.league_agents_t2.append(Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device))

            self.main_agents_t2_counter = [0] * self.args.n_main_agents
            self.coaching_agents_t2_counter = [0] * self.args.n_coaching_agents
            self.league_agents_t2_counter = [0] * self.args.n_league_agents

    def select_opponent(self, 
                        agent_type, 
                        main_agents,
                        coaching_agents,
                        league_agents,
                        historical_agents,
                        main_agents_prev,
                        coaching_agents_prev,
                        league_agents_prev
                        ):
        """
        Select agents for training.
        """
        opponent_types = ['main', 'coach', 'league', 'historical']
        weights = self.opponent_selection_weights[agent_type]

        # Adjust weights if we do not have any historical agents yet
        if len(historical_agents) == 0:
            weights = (weights[0]/sum(weights[:3]), weights[1]/sum(weights[:3]), weights[2]/sum(weights[:3]), 0.0)
            
        # Choose opponent
        opponent_type = np.random.choice(opponent_types, p=weights)
        opponent_idx = np.random.choice(np.arange(len(main_agents)) if opponent_type == 'main' else
                                        np.arange(len(coaching_agents)) if opponent_type == 'coach' else
                                        np.arange(len(league_agents)) if opponent_type == 'league' else
                                        np.arange(len(historical_agents)))
                                
        # Train against the PREVIOUS version of the opponent 
        if opponent_type == 'main':
            opponent = main_agents_prev[opponent_idx]
        elif opponent_type == 'coach':
            opponent = coaching_agents_prev[opponent_idx]
        elif opponent_type == 'league':
            opponent = league_agents_prev[opponent_idx]
        else:
            opponent = historical_agents[opponent_idx]

        return opponent, opponent_type, opponent_idx
    
    def train_agent(self, 
                    iteration, 
                    agent_pool, 
                    agent_counter, 
                    agent_type, 
                    main_agents,
                    coaching_agents,
                    league_agents,
                    historical_agents,
                    main_agents_prev,
                    coaching_agents_prev,
                    league_agents_prev, 
                    train_team1=True
                    ):
        """
        Train agents.
        """
        for agent_idx, agent in enumerate(agent_pool):
            opponent, opponent_type, opponent_idx = self.select_opponent(agent_type,
                                                                        main_agents,
                                                                        coaching_agents,
                                                                        league_agents,
                                                                        historical_agents,
                                                                        main_agents_prev,
                                                                        coaching_agents_prev,
                                                                        league_agents_prev)

            team_idx = 0 if train_team1 else 1
            print(f'Iteration: {iteration} Team: {team_idx} Training {agent_type} agent {agent_idx} vs {opponent_type} agent {opponent_idx}')

            # Train PPO
            total_rewards = self.ppotrainer.train_ppo(self.args, self.env, agent, opponent, train_team1=train_team1)
            clear_output()

            # Store rewards
            self.learning_rewards[iteration][agent_type + str(agent_idx) + str(team_idx)] = total_rewards

            # Increment counter
            agent_counter[agent_idx] += 1


    def train_all_agents_symmetric(self, iteration=0):
        # Create copies of previous agent states to train against
        self.main_agents_t1_prev = [copy.deepcopy(model) for model in self.main_agents_t1]
        self.coaching_agents_t1_prev = [copy.deepcopy(model) for model in self.coaching_agents_t1]
        self.league_agents_t1_prev = [copy.deepcopy(model) for model in self.league_agents_t1]

        # Train main agents
        self.train_agent(iteration, 
                        self.main_agents_t1, 
                        self.main_agents_t1_counter, 
                        'main', 
                        self.main_agents_t1,
                        self.coaching_agents_t1,
                        self.league_agents_t1,
                        self.historical_agents_t1,
                        self.main_agents_t1_prev,
                        self.coaching_agents_t1_prev,
                        self.league_agents_t1_prev, 
                        train_team1=True)
        
        # Train coaching agents
        self.train_agent(iteration, 
                        self.coaching_agents_t1, 
                        self.coaching_agents_t1_counter, 
                        'coach', 
                        self.main_agents_t1,
                        self.coaching_agents_t1,
                        self.league_agents_t1,
                        self.historical_agents_t1,
                        self.main_agents_t1_prev,
                        self.coaching_agents_t1_prev,
                        self.league_agents_t1_prev, 
                        train_team1=True)
        
        # Train league agents
        self.train_agent(iteration, 
                        self.league_agents_t1, 
                        self.league_agents_t1_counter, 
                        'league', 
                        self.main_agents_t1,
                        self.coaching_agents_t1,
                        self.league_agents_t1,
                        self.historical_agents_t1,
                        self.main_agents_t1_prev,
                        self.coaching_agents_t1_prev,
                        self.league_agents_t1_prev, 
                        train_team1=True)
       
    def train_all_agents_non_symmetric(self, iteration=0):
        # Create copies of previous agent states to train against
        self.main_agents_t1_prev = [copy.deepcopy(model) for model in self.main_agents_t1]
        self.coaching_agents_t1_prev = [copy.deepcopy(model) for model in self.coaching_agents_t1]
        self.league_agents_t1_prev = [copy.deepcopy(model) for model in self.league_agents_t1]

        # Create copies of previous agent states to train against
        self.main_agents_t2_prev = [copy.deepcopy(model) for model in self.main_agents_t2]
        self.coaching_agents_t2_prev = [copy.deepcopy(model) for model in self.coaching_agents_t2]
        self.league_agents_t2_prev = [copy.deepcopy(model) for model in self.league_agents_t2]

        # Train main agents
        self.train_agent(iteration, 
                        self.main_agents_t1, 
                        self.main_agents_t1_counter, 
                        'main', 
                        self.main_agents_t2,
                        self.coaching_agents_t2,
                        self.league_agents_t2,
                        self.historical_agents_t2,
                        self.main_agents_t2_prev,
                        self.coaching_agents_t2_prev,
                        self.league_agents_t2_prev, 
                        train_team1=True)
        
        self.train_agent(iteration, 
                        self.main_agents_t2, 
                        self.main_agents_t2_counter, 
                        'main', 
                        self.main_agents_t1,
                        self.coaching_agents_t1,
                        self.league_agents_t1,
                        self.historical_agents_t1,
                        self.main_agents_t1_prev,
                        self.coaching_agents_t1_prev,
                        self.league_agents_t1_prev, 
                        train_team1=False)
        
        # Train coaching agents
        self.train_agent(iteration, 
                        self.coaching_agents_t1, 
                        self.coaching_agents_t1_counter, 
                        'coach', 
                        self.main_agents_t2,
                        self.coaching_agents_t2,
                        self.league_agents_t2,
                        self.historical_agents_t2,
                        self.main_agents_t2_prev,
                        self.coaching_agents_t2_prev,
                        self.league_agents_t2_prev, 
                        train_team1=True)
        
        self.train_agent(iteration, 
                        self.coaching_agents_t2, 
                        self.coaching_agents_t2_counter, 
                        'coach',
                        self.main_agents_t1,
                        self.coaching_agents_t1,
                        self.league_agents_t1,
                        self.historical_agents_t1,
                        self.main_agents_t1_prev,
                        self.coaching_agents_t1_prev,
                        self.league_agents_t1_prev, 
                        train_team1=False)
        
        # Train league agents
        self.train_agent(iteration, 
                        self.league_agents_t1, 
                        self.league_agents_t1_counter, 
                        'league', 
                        self.main_agents_t2,
                        self.coaching_agents_t2,
                        self.league_agents_t2,
                        self.historical_agents_t2,
                        self.main_agents_t2_prev,
                        self.coaching_agents_t2_prev,
                        self.league_agents_t2_prev, 
                        train_team1=True)
        
        self.train_agent(iteration, 
                        self.league_agents_t2, 
                        self.league_agents_t2_counter, 
                        'league',
                        self.main_agents_t1,
                        self.coaching_agents_t1,
                        self.league_agents_t1,
                        self.historical_agents_t1,
                        self.main_agents_t1_prev,
                        self.coaching_agents_t1_prev,
                        self.league_agents_t1_prev, 
                        train_team1=False)
            
    def calculate_winrate_matrix_symmetric(self):
        """
        Calculate winrate matrix when teams are symmetric.
        """

        winrate_matrix = defaultdict(int)
        draw_matrix = defaultdict(int)
        t1_label = '0_'

        # Parallelise duelling
        async_results = []
        for agent_idx, agent in enumerate(self.all_agents_t1):
            for opponent_idx, opponent in enumerate(self.all_agents_t1):
                if agent_idx > opponent_idx:
                    for _ in range(self.args.number_of_duels):
                        idxs = (t1_label + str(agent_idx), t1_label + str(opponent_idx))
                        async_result = ray_duel.remote(self.env, agent, opponent, idxs, return_result=True, device=self.args.device)
                        async_results.append(async_result)
        duel_results = ray.get(async_results)

        # Calculate win rates
        for agent_idx, opponent_idx, result in duel_results:
            winrate_matrix[(agent_idx, opponent_idx)] += (result == 1) / self.args.number_of_duels
            draw_matrix[(agent_idx, opponent_idx)] += (result == 0) / self.args.number_of_duels
        
        # Get inverse win rates
        keys = list(winrate_matrix.keys())
        for agent_idx, opponent_idx in keys:
            winrate_matrix[(opponent_idx, agent_idx)] = 1 - winrate_matrix[(agent_idx, opponent_idx)] - draw_matrix[(agent_idx, opponent_idx)]

        self.winrate_matrix = winrate_matrix
        self.winrate_matrices.append(winrate_matrix)

    def create_all_agents_list(self):
        """
        Create lists of all agents.
        """

        self.all_agents_t1 = self.main_agents_t1 \
                        + self.coaching_agents_t1 \
                        + self.league_agents_t1 \
                        + self.historical_agents_t1
        
        self.all_agents_t1_counters = self.main_agents_t1_counter \
                        + self.coaching_agents_t1_counter \
                        + self.league_agents_t1_counter \
                        + [np.inf] * len(self.historical_agents_t1)

        self.all_agents_t2 = []      
        if not self.symmetric_teams:
            self.all_agents_t2 = self.main_agents_t2 \
                            + self.coaching_agents_t2 \
                            + self.league_agents_t2 \
                            + self.historical_agents_t2
            
            self.all_agents_t2_counters = self.main_agents_t2_counter \
                            + self.coaching_agents_t2_counter \
                            + self.league_agents_t2_counter \
                            + [np.inf] * len(self.historical_agents_t2)

    def calculate_winrate_matrix_non_symmetric(self):
        """
        Calculate winrate matrix when teams are not symmetric.
        """

        winrate_matrix = defaultdict(int)
        draw_matrix = defaultdict(int)
        t1_label = '0_'
        t2_label = '1_'

        # Parallelise duelling
        async_results = []
        for agent_idx, agent in enumerate(self.all_agents_t1):
            for opponent_idx, opponent in enumerate(self.all_agents_t2):
                for _ in range(self.args.number_of_duels):
                    idxs = (t1_label + str(agent_idx), t2_label + str(opponent_idx))
                    async_result = ray_duel.remote(self.env, agent, opponent, idxs, return_result=True, device=self.args.device)
                    async_results.append(async_result)
        duel_results = ray.get(async_results)

        # Calculate win rates
        for agent_idx, opponent_idx, result in duel_results:
            winrate_matrix[(agent_idx, opponent_idx)] += (result == 1) / self.args.number_of_duels
            draw_matrix[(agent_idx, opponent_idx)] += (result == 0) / self.args.number_of_duels
        
        # Get inverse win rates
        keys = list(winrate_matrix.keys())
        for agent_idx, opponent_idx in keys:
            winrate_matrix[(opponent_idx, agent_idx)] = 1 - winrate_matrix[(agent_idx, opponent_idx)] - draw_matrix[(agent_idx, opponent_idx)]

        self.winrate_matrix = winrate_matrix
        self.winrate_matrices.append(winrate_matrix)
    
    def update_agent_pools(self, team_idx=0):
        """
        Update agent pools when teams are symmetric.
        """

        print(f'Updating agent pools for team {team_idx}')

        keys = list(self.winrate_matrix.keys())
        agent_labels = list(set(k[0] for k in keys if int(k[0][0])==team_idx))
        agent_idxs = np.arange(len(agent_labels))
        wr_items = list(self.winrate_matrix.items())
        agent_avg_win_rates = np.array([np.mean([v for k, v in wr_items if k[0] == i]) for i in agent_labels])

        if team_idx == 0:
            self.agent_avg_win_rates_t1 = agent_avg_win_rates
        else:
            self.agent_avg_win_rates_t2 = agent_avg_win_rates
        print(f'Agent win rates: {agent_avg_win_rates}')
        
        # Break ties randomly if multiple agents have the same winrate
        best_agent_idx = np.random.choice(np.flatnonzero(agent_avg_win_rates == agent_avg_win_rates.max()))
        
        best_agent_win_rate = agent_avg_win_rates[best_agent_idx]
        if team_idx == 0:
            best_agent_weights = self.all_agents_t1[best_agent_idx].state_dict()
        else:
            best_agent_weights = self.all_agents_t2[best_agent_idx].state_dict()
            
        print(f'Best agent: {best_agent_idx}, win rate of {best_agent_win_rate}')

        # Remove underperforming historical agents
        historical_agents_to_remove = []
        if team_idx == 0:
            n_historical_agents = len(self.historical_agents_t1)
        else:
            n_historical_agents = len(self.historical_agents_t2)

        for agent_idx in np.arange(n_historical_agents):
            adj_agent_idx = agent_idx + self.args.n_main_agents + self.args.n_coaching_agents + self.args.n_league_agents
            if agent_avg_win_rates[adj_agent_idx] < self.args.min_historical_agent_winrate:
                historical_agents_to_remove.append(agent_idx)
                print(f'Historical agents for removal: {agent_idx}, winrate of {agent_avg_win_rates[adj_agent_idx]}')
            if team_idx == 0:
                self.historical_agents_t1 = [v for agent_idx, v in enumerate(self.historical_agents_t1) if agent_idx not in historical_agents_to_remove]
            else:
                self.historical_agents_t2 = [v for agent_idx, v in enumerate(self.historical_agents_t2) if agent_idx not in historical_agents_to_remove]

        # Replace agents if best agent is good enough
        if best_agent_win_rate > self.args.min_agent_winrate_for_promotion:
            for agent_idx in agent_idxs[:self.args.n_main_agents + self.args.n_coaching_agents + self.args.n_league_agents]:
                if team_idx == 0:
                    agent_iteration_count = self.all_agents_t1_counters[agent_idx]
                else:
                    agent_iteration_count = self.all_agents_t2_counters[agent_idx]
                if agent_avg_win_rates[agent_idx] < self.args.min_agent_winrate and agent_iteration_count > self.args.min_agent_iterations_for_replacement:
                    if agent_idx < self.args.n_main_agents:
                        adj_agent_idx = agent_idx
                        replacement_agent = Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device)
                        replacement_agent.load_state_dict(best_agent_weights)
                        if team_idx == 0:
                            self.main_agents_t1[adj_agent_idx] = replacement_agent
                            self.main_agents_t1_counter[adj_agent_idx] = 0
                        else:
                            self.main_agents_t2[adj_agent_idx] = replacement_agent
                            self.main_agents_t2_counter[adj_agent_idx] = 0
                        print(f'Replacing main agent {adj_agent_idx} with agent {best_agent_idx}')

                    elif agent_idx < (self.args.n_main_agents + self.args.n_coaching_agents):
                        adj_agent_idx = agent_idx - self.args.n_main_agents
                        replacement_agent = Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device)
                        replacement_agent.load_state_dict(best_agent_weights)
                        if team_idx == 0:
                            self.coaching_agents_t1[adj_agent_idx] = replacement_agent
                            self.coaching_agents_t1_counter[adj_agent_idx] = 0
                        else:
                            self.coaching_agents_t2[adj_agent_idx] = replacement_agent
                            self.coaching_agents_t2_counter[adj_agent_idx] = 0                           
                        print(f'Replacing coaching agent {adj_agent_idx} with agent {best_agent_idx}')

                    elif agent_idx < (self.args.n_main_agents + self.args.n_coaching_agents + self.args.n_league_agents):
                        adj_agent_idx = agent_idx - (self.args.n_main_agents + self.args.n_coaching_agents)
                        replacement_agent = Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device)
                        replacement_agent.load_state_dict(best_agent_weights)
                        if team_idx == 0:
                            self.league_agents_t1[adj_agent_idx] = replacement_agent
                            self.league_agents_t1_counter[adj_agent_idx] = 0
                        else:
                            self.league_agents_t2[adj_agent_idx] = replacement_agent
                            self.league_agents_t2_counter[adj_agent_idx] = 0
                        print(f'Replacing league agent {adj_agent_idx} with agent {best_agent_idx}')

            # Add best performing agent to historical agent pool
            print(f'Updating historical agent pool')
            if team_idx == 0:
                historical_agent_weights = [str(a.state_dict()) for a in self.historical_agents_t1]
            else:
                historical_agent_weights = [str(a.state_dict()) for a in self.historical_agents_t2]
            if str(best_agent_weights) in historical_agent_weights:
                print('Best agent already exists in historical agent pool!')
            else:
                print(f'Adding agent {best_agent_idx} to historic agents\n')
                new_agent = Agent(self.args.n_actions, self.n_channels, self.env.GRID_SIZE, self.local_metadata_dims[0]).to(self.args.device)
                new_agent.load_state_dict(best_agent_weights)
                if team_idx == 0:
                    self.historical_agents_t1.append(new_agent)
                else:
                    self.historical_agents_t2.append(new_agent)
                if team_idx == 0 and len(self.historical_agents_t1) > self.args.n_historical_agents:
                    self.historical_agents_t1.pop(0)
                elif len(self.historical_agents_t2) > self.args.n_historical_agents:
                    self.historical_agents_t2.pop(0)
            
    def generate_metrics_symmetric(self, iteration, env_copy):
        """
        Generate metrics for symmetric teams.
        """

        agent_labels = ['m' + str(i) for i in range(self.args.n_main_agents)] \
                     + ['c' + str(i) for i in range(self.args.n_coaching_agents)] \
                     + ['l' + str(i) for i in range(self.args.n_league_agents)] 
        
        idxs_to_keep = len(self.all_agents_t1) - len(self.historical_agents_t1)
        scaling_factor = 1 / (len(self.all_agents_t1) - 1 + self.args.number_of_duels)

        # Use the remote function to run the 'duel' method in parallel
        async_results = []
        number_of_duels = 0
        for agent_idx, agent in enumerate(self.all_agents_t1[:idxs_to_keep]):
            for opponent_idx, opponent in enumerate(self.all_agents_t1[:idxs_to_keep]):
                for _ in range(self.args.number_of_duels):
                    number_of_duels += 1
                    idxs = (agent_idx, opponent_idx)
                
                    async_result = ray_duel.remote(env_copy, agent, opponent, idxs, return_result=False, device=self.args.device)
                    async_results.append(async_result)

        # Collect the results
        duel_results = ray.get(async_results)

        for agent_idx, opponent_idx, metrics in duel_results:
            agent_label = agent_labels[agent_idx]
            self.metlog.harvest_metrics(metrics, agent_label, iteration, scaling_factor)

        print(f'Total duels {number_of_duels}')

    def generate_metrics_non_symmetric(self, iteration, env_copy):
        """
        Generate metrics for symmetric teams.
        """

        agent_labels_t1 = ['t0_m' + str(i) for i in range(self.args.n_main_agents)] \
                        + ['t0_c' + str(i) for i in range(self.args.n_coaching_agents)] \
                        + ['t0_l' + str(i) for i in range(self.args.n_league_agents)] \
                        + ['t0_h' + str(i) for i in range(len(self.historical_agents_t1))] 
        
        agent_labels_t2 = ['t1_m' + str(i) for i in range(self.args.n_main_agents)] \
                        + ['t1_c' + str(i) for i in range(self.args.n_coaching_agents)] \
                        + ['t1_l' + str(i) for i in range(self.args.n_league_agents)] \
                        + ['t1_h' + str(i) for i in range(len(self.historical_agents_t1))] 
        
        idxs_to_keep_t1 = len(self.all_agents_t1) - len(self.historical_agents_t1)
        idxs_to_keep_t2 = len(self.all_agents_t2) - len(self.historical_agents_t2)

        scaling_factor_t1 = 1 / (len(self.all_agents_t2) - 1 + self.args.number_of_duels)
        scaling_factor_t2 = 1 / (len(self.all_agents_t1) - 1 + self.args.number_of_duels)
        
        # Use the remote function to run the 'duel' method in parallel
        async_results = []
        number_of_duels = 0
        for agent_idx, agent in enumerate(self.all_agents_t1[:idxs_to_keep_t1]):
            for opponent_idx, opponent in enumerate(self.all_agents_t2[:idxs_to_keep_t2]):
                for _ in range(self.args.number_of_duels):
                    number_of_duels += 1
                    idxs = (agent_idx, opponent_idx)
                
                    async_result = ray_duel.remote(env_copy, agent, opponent, idxs, return_result=False, device=self.args.device)
                    async_results.append(async_result)

        # Collect the results
        duel_results_t1 = ray.get(async_results)

        for agent_idx, opponent_idx, metrics in duel_results_t1:
            agent_t1_label = agent_labels_t1[agent_idx]
            agent_t2_label = agent_labels_t2[opponent_idx]
            self.metlog.harvest_metrics(metrics, agent_t1_label, iteration, scaling_factor_t1, team_idx=0)
            self.metlog.harvest_metrics(metrics, agent_t2_label, iteration, scaling_factor_t2, team_idx=1)

        print(f'Total duels {number_of_duels}')

    def checkpoint(self, meta_run, iteration):
        """
        Save down objects.
        """
        iteration_adj = iteration + 1

        time_capsule = TimeCapsule(
                            self.args,
                            self.symmetric_teams,
                            self.env.AGENT_TYPES,
                            self.env.AGENT_TEAMS,
                            self.all_agents_t1,
                            self.all_agents_t2,
                            self.metlog,
                            self.winrate_matrices)

        if not os.path.exists(self.OUTPUT_FOLDER_NAME):
            os.makedirs(self.OUTPUT_FOLDER_NAME)

        with open(self.OUTPUT_FOLDER_NAME + '/' + self.TIME_CAPSULE_FILE_NAME + '_' + str(meta_run) + '_' + str(iteration_adj) + self.FILE_EXT, 'wb') as f:
            pickle.dump(time_capsule, f)

    def train_league(self):
        """
        Train the league.
        """
        
        if self.args.use_wandb_selfplay:
            wandb.init(project=self.args.wandb_project_name,
                        name=self.args.exp_name,
                        config=vars(self.args))
                        
        # Init ray for parallelisation
        ray.shutdown() 
        ray.init()
        
        env_id = ray.put(self.env)
        env_copy = ray.get(env_id)

        #----------------------------------------------------------------------
        # League Training Start
        #----------------------------------------------------------------------
        for meta_run in range(self.args.number_of_metaruns):
            self._init_objects()

            for iteration in range(self.args.number_of_iterations):

                #----------------------------------------------------------------------
                # Train agents
                #----------------------------------------------------------------------
                print(f'Meta run: {meta_run} Iteration: {iteration} Training agents...')
                if not self.symmetric_teams:
                    print('Teams are not symmetric - two leagues will be trained')
                start_time = time.perf_counter()

                if self.symmetric_teams:
                    self.train_all_agents_symmetric(iteration)
                else:
                    self.train_all_agents_non_symmetric(iteration)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f'Training agents took {elapsed_time:.4f} seconds to run.\n')

                #----------------------------------------------------------------------
                # Duelling Phase and harvest of metrics
                #----------------------------------------------------------------------

                self.create_all_agents_list()

                if iteration % self.args.inference_interval == 0:
                    print(f'Meta run: {meta_run} Iteration: {iteration} Metrics collection...')
                    start_time = time.perf_counter()
                    
                    if len(self.all_agents_t1) >= 1:
                        if self.symmetric_teams:
                            self.generate_metrics_symmetric(iteration, env_copy)
                        else:
                            self.generate_metrics_non_symmetric(iteration, env_copy)

                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    print(f'Metrics collection took {elapsed_time:.4f} seconds to run.\n') 

                # Log metrics
                if self.args.use_wandb_selfplay:
                    self.metlog.log_to_wandb()

                #----------------------------------------------------------------------
                # Calculate overall win matrix
                #----------------------------------------------------------------------
                print(f'Meta run: {meta_run} Iteration: {iteration} Calculating win rate matrix...')
                start_time = time.perf_counter()

                if len(self.all_agents_t1) >= 1:
                    if self.symmetric_teams:
                        self.calculate_winrate_matrix_symmetric()
                    else:
                        self.calculate_winrate_matrix_non_symmetric()

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f'Calculating winrate matrix took {elapsed_time:.4f} seconds to run.\n')

                #----------------------------------------------------------------------
                # Update agent pools
                #----------------------------------------------------------------------
                print(f'Meta run: {meta_run} Iteration: {iteration} updating agent pools...')
                start_time = time.perf_counter()

                if iteration > self.args.min_learning_rounds \
                  and len(self.all_agents_t1) >= 1 and len(self.all_agents_t2) >= 1:
                    self.update_agent_pools(team_idx=0)
                    if not self.symmetric_teams:
                        self.update_agent_pools(team_idx=1)

                    print(self.winrate_matrix)
                        
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f'Updating agent pools took {elapsed_time:.4f} seconds to run.\n')

                #----------------------------------------------------------------------
                # Checkpointing
                #----------------------------------------------------------------------
                if iteration == 0 or (iteration + 1) % self.args.checkpoint_frequency == 0:
                    print(f'Saving objects...\n')
                    self.checkpoint(meta_run, iteration)

            # Close wandb session
            if self.args.use_wandb_selfplay:
                self.metlog.log_matplotlib_plots_to_wandb()
                self.metlog.log_summary_plot_to_wandb()
                # metlog.log_wandb_table_plots()
                wandb.finish()

            # Shutdown Ray
            ray.shutdown()
