import wandb
from collections import defaultdict
import copy
import matplotlib.pyplot as plt

class MetricsLogger:
    """
    Class to log metrics
    """
    def __init__(self, n_agents, use_wandb=False):

        self.n_agents = n_agents
        self.use_wandb = use_wandb
        self.team_metrics = {
            # Team level metrics
            "team_total_tag_count": [],
            "team_respawn_tag_count": [],
            "team_flag_captures": [],
            "team_blocks_laid": [],
            "team_blocks_mined": [],
            "team_total_distance_to_own_flag": [],
            "team_total_distance_to_opp_flag": [],
            "team_health_pickups": []
        }
        self.agent_type_metrics = {
            # Agent type metrics
            "agent_type_total_tag_count": defaultdict(list),
            "agent_type_respawn_tag_count": defaultdict(list),
            "agent_type_flag_captures": defaultdict(list),
            "agent_type_blocks_laid": defaultdict(list),
            "agent_type_blocks_mined": defaultdict(list),
            "agent_type_total_distance_to_own_flag": defaultdict(list),
            "agent_type_total_distance_to_opp_flag": defaultdict(list),
            "agent_type_health_pickups": defaultdict(list)
        }

        self.metrics = {
            i:{**copy.deepcopy(self.team_metrics), **copy.deepcopy(self.agent_type_metrics)} 
            for i in range(self.n_agents)
        }

        self.current_step = 0

        # self.wandb_capture_table = wandb.Table(columns=["agent", "value", "step"])

    def _init_metric_list(self,env, step):
        """
        Initialise metrics lists for the current step.
        """
        for k1 in self.metrics.keys():
            for k2 in self.team_metrics.keys():
                if len(self.metrics[k1][k2]) < step + 1:
                    self.metrics[k1][k2].append(0)

        for k1 in self.metrics.keys():
            for k2 in self.agent_type_metrics.keys():
                for k3 in set(env.AGENT_TYPES.values()):
                    if len(self.metrics[k1][k2][k3]) < step + 1:
                        self.metrics[k1][k2][k3].append(0)

    def harvest_metrics(self, env, agent_idx, step, scaling_factor):
        """
        Harvest metrics.
        """
        self.current_step = step

        # Init metrics lists for the current step
        self._init_metric_list(env, step)
        
        # Populate metrics with data stored in env
        for k2 in self.team_metrics.keys():
            # metrics[metric][team]
            self.metrics[agent_idx][k2][step] += env.metrics[k2][0] * scaling_factor

        for k2 in self.agent_type_metrics.keys():
            for k3 in set(env.AGENT_TYPES.values()):
                # metrics[metric][team][agent_type]
                self.metrics[agent_idx][k2][k3][step] += env.metrics[k2][0][k3] * scaling_factor
    
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