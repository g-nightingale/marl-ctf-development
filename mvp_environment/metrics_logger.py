import wandb
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import numpy as np

class MetricsLogger:
    """
    Class to log metrics
    """
    def __init__(self, 
                 n_main_agents, 
                 n_coaching_agents,
                 n_league_agents,
                 agent_team_types,
                 agent_indiv_idxs,
                 symmetric_teams,
                 use_wandb=False):

        self.n_main_agents = n_main_agents
        self.n_coaching_agents = n_coaching_agents
        self.n_league_agents = n_league_agents
        self.agent_team_types = agent_team_types
        self.agent_indiv_idxs = agent_indiv_idxs
        self.symmetric_teams = symmetric_teams
        self.use_wandb = use_wandb
        self.team_metrics = {
            # Team level metrics
            "team_tag_count": [],
            "team_respawn_tag_count": [],
            "team_flag_pickups": [],
            "team_flag_captures": [],
            "team_flag_dispossessions": [],
            "team_blocks_laid": [],
            "team_blocks_mined": [],
            "team_blocks_laid_distance_from_own_flag": [],
            "team_blocks_laid_distance_from_opp_flag": [],
            "team_steps_defending_zone": [],
            "team_steps_attacking_zone": [],
            "team_steps_adj_teammate": [],
            "team_steps_adj_opponent": []
        }
        self.agent_type_metrics = {
            # Agent type metrics
            "agent_type_tag_count": defaultdict(list),
            "agent_type_respawn_tag_count": defaultdict(list),
            "agent_type_flag_pickups": defaultdict(list),
            "agent_type_flag_captures": defaultdict(list),
            "agent_type_flag_dispossessions": defaultdict(list),
            "agent_type_blocks_laid": defaultdict(list),
            "agent_type_blocks_mined": defaultdict(list),
            "agent_type_blocks_laid_distance_from_own_flag": defaultdict(list),
            "agent_type_blocks_laid_distance_from_opp_flag": defaultdict(list),
            "agent_type_steps_defending_zone": defaultdict(list),
            "agent_type_steps_attacking_zone": defaultdict(list),
            "agent_type_steps_adj_teammate": defaultdict(list),
            "agent_type_steps_adj_opponent": defaultdict(list)
        }
        self.agent_metrics = {
            # Agent type metrics
            "agent_tag_count": defaultdict(list),
            "agent_respawn_tag_count": defaultdict(list),
            "agent_flag_pickups": defaultdict(list),
            "agent_flag_captures": defaultdict(list),
            "agent_flag_dispossessions": defaultdict(list),
            "agent_blocks_laid": defaultdict(list),
            "agent_blocks_mined": defaultdict(list),
            "agent_blocks_laid_distance_from_own_flag": defaultdict(list),
            "agent_blocks_laid_distance_from_opp_flag": defaultdict(list),
            "agent_steps_defending_zone": defaultdict(list),
            "agent_steps_attacking_zone": defaultdict(list),
            "agent_steps_adj_teammate": defaultdict(list),
            "agent_steps_adj_opponent": defaultdict(list)
        }

        if self.symmetric_teams:
            agent_labels = ['m' + str(i) for i in range(self.n_main_agents)] \
                         + ['c' + str(i) for i in range(self.n_coaching_agents)] \
                         + ['l' + str(i) for i in range(self.n_league_agents)] 
        else:
            agent_labels = ['t0_m' + str(i) for i in range(self.n_main_agents)] \
                         + ['t0_c' + str(i) for i in range(self.n_coaching_agents)] \
                         + ['t0_l' + str(i) for i in range(self.n_league_agents)] \
                         + ['t1_m' + str(i) for i in range(self.n_main_agents)] \
                         + ['t1_c' + str(i) for i in range(self.n_coaching_agents)] \
                         + ['t1_l' + str(i) for i in range(self.n_league_agents)] 
        
        self.metrics = {
                i:{**copy.deepcopy(self.team_metrics), 
                   **copy.deepcopy(self.agent_type_metrics),
                   **copy.deepcopy(self.agent_metrics)
                   } 
                for i in agent_labels
        }
     
        self.current_step = 0

        # self.wandb_capture_table = wandb.Table(columns=["agent", "value", "step"])

    def _init_metric_list(self, step, team_idx):
        """
        Initialise metrics lists for the current step.
        """
        for agent_idx in self.metrics.keys():
            for metric in self.team_metrics.keys():
                metric_len = len(self.metrics[agent_idx][metric])
                if metric_len < step + 1:
                    if metric_len > 0:
                        # Fill any gaps with most recent value
                        for _ in range(metric_len+1, step+1):
                            self.metrics[agent_idx][metric].append(self.metrics[agent_idx][metric][metric_len-1])   
                    # Initialise new array element            
                    self.metrics[agent_idx][metric].append(0.0)

            for metric in self.agent_type_metrics.keys():
                for agent_type in self.agent_team_types[team_idx]:
                    metric_len = len(self.metrics[agent_idx][metric][agent_type])
                    if metric_len < step + 1:
                        if metric_len > 0:
                            # Fill any gaps with most recent value
                            for _ in range(metric_len+1, step+1):
                                self.metrics[agent_idx][metric][agent_type].append(self.metrics[agent_idx][metric][agent_type][metric_len-1])               
                        # Initialise new array element
                        self.metrics[agent_idx][metric][agent_type].append(0.0)

            for metric in self.agent_metrics.keys():
                for agent_indiv_idx in self.agent_indiv_idxs:
                    metric_len = len(self.metrics[agent_idx][metric][agent_indiv_idx])
                    if metric_len < step + 1:
                        if metric_len > 0:
                            # Fill any gaps with most recent value
                            for _ in range(metric_len+1, step+1):
                                self.metrics[agent_idx][metric][agent_indiv_idx].append(self.metrics[agent_idx][metric][agent_indiv_idx][metric_len-1])               
                        # Initialise new array element
                        self.metrics[agent_idx][metric][agent_indiv_idx].append(0.0)

    def harvest_metrics(self, metrics, agent_idx, step, scaling_factor, team_idx=0):
        """
        Harvest metrics.
        """
        self.current_step = step

        # Init metrics lists for the current step
        self._init_metric_list(step, team_idx)
        
        # Populate metrics with data stored in env
        for metric in self.team_metrics.keys():
            # metrics[metric][team]
            self.metrics[agent_idx][metric][step] += metrics[metric][team_idx] * scaling_factor

        for metric in self.agent_type_metrics.keys():
            for agent_type in self.agent_team_types[team_idx]:
                # metrics[metric][team][agent_type]
                self.metrics[agent_idx][metric][agent_type][step] += metrics[metric][team_idx][agent_type] * scaling_factor
    
        for metric in self.agent_metrics.keys():
            for agent_indiv_idx in self.agent_indiv_idxs:
                # metrics[metric][team][agent_type]
                self.metrics[agent_idx][metric][agent_indiv_idx][step] += metrics[metric][agent_indiv_idx] * scaling_factor

    def log_wandb_table_plots(self):
        """
        Log wandb table plots.
        """

        flag_captures_plot = wandb.plot_table("wandb/line/v0", 
                                            self.wandb_capture_table, 
                                            {"x": "step", "y": "value", "groupKeys": "agent"}, 
                                            {"title": "Flag Captures"}
                                            )


        wandb.log({"flag_captures_tbl": flag_captures_plot})
            
    def log_to_wandb(self, step=None):
        """
        Log metrics to weights and biases.
        """

        if step is None:
            step = self.current_step

        # Collect metrics
        metrics_step = {}
        for agent in self.metrics.keys():
            for metric in self.team_metrics.keys():
                metric_name = f'{metric} (agent:{agent})'
                metric_value = self.metrics[agent][metric][step]
                metrics_step[metric_name] = metric_value

            for metric in self.agent_type_metrics.keys():
                for agent_type in self.metrics[agent][metric].keys():
                    metric_name = f'{metric} (agent:{agent}, agent type:{agent_type})'
                    metric_value = self.metrics[agent][metric][agent_type][step]
                    metrics_step[metric_name] = metric_value

            for metric in self.agent_metrics.keys():
                for agent_indiv_idx in self.metrics[agent][metric].keys():
                    metric_name = f'{metric} (agent:{agent}, agent individual:{agent_indiv_idx})'
                    metric_value = self.metrics[agent][metric][agent_indiv_idx][step]
                    metrics_step[metric_name] = metric_value

        # Log to wandb
        wandb.log(metrics_step)

    def log_matplotlib_plots_to_wandb(self):
        """
        Log plotly charts to wandb.
        """
        STYLE = 'seaborn-v0_8-whitegrid'
        agents = self.metrics.keys()

        # Plot team metrics
        for metric in self.team_metrics.keys():
            # Get the data
            plt_data = {agent:self.metrics[agent][metric] for agent in agents}

            # Plot the data
            plt.title(metric)
            plt.style.use(STYLE)
            for d in plt_data.keys():
                plt.plot(plt_data[d], label=d)
            plt.legend()
            plt.ylabel(metric)
            plt.xlabel('iteration')

            # Log to wandb
            wandb.log({metric: plt})

        # Plot agent type metrics
        for metric in self.agent_type_metrics.keys():
            for agent_type in self.metrics[0][metric].keys():
                metric_name = f'{metric} (agent type:{agent_type})'
                plt_data = {agent:self.metrics[agent][metric][agent_type] for agent in agents}   

                # Plot the data
                plt.title(metric_name)
                for d in plt_data.keys():
                    plt.plot(plt_data[d], label=d)
                plt.legend()
                plt.ylabel(metric)
                plt.xlabel('iteration')

                # Log to wandb
                wandb.log({metric_name: plt})