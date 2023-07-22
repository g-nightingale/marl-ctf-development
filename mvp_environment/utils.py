# Plot rewards
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pickle 
import torch
import os
import pickle
import imageio
from scipy import stats
import pandas as pd
from collections import defaultdict
import ray
import copy

class TrainingConfig():
    def __init__(self):
        self.check = 1

def get_agents(run_dir,
                file_name,
                agent1_team,
                agent1_idx,
                agent2_team,
                agent2_idx):
    
    """
    Recovers saved agents from time capsule.
    """
    
    file_path = run_dir + file_name
    print(file_path)

    with open(file_path, 'rb') as f:
        time_capsule = pickle.load(f)

    env_config = time_capsule.args.env_config

    if agent1_team == 0:
        agent1 = time_capsule.all_agents_t1[agent1_idx]
    else:
        agent1 = time_capsule.all_agents_t2[agent1_idx]

    if agent2_team == 0:
        agent2 = time_capsule.all_agents_t1[agent2_idx]
    else:
        agent2 = time_capsule.all_agents_t2[agent2_idx]

    return env_config, agent1, agent2

def create_plots(run_dir,
                file_names,
                include_block_tile_metrics=True):
    """
    Create primary plots for analysis.
    """

    if include_block_tile_metrics:
        NROWS = 5
        NCOLS = 3
        FIGSIZE = (10, 10)
        N_METRICS = 13
    else:
        NROWS = 3
        NCOLS = 3
        FIGSIZE = (10, 6)
        N_METRICS = 9

    # Load data
    time_capsules = []
    for file_name in file_names:
        file_path = run_dir + file_name
        print(file_path)
    
        with open(file_path, 'rb') as f:
            time_capsules.append(pickle.load(f))

    team_metrics = [
                "team_flag_pickups",
                "team_flag_captures",
                "team_tag_count",
                "team_respawn_tag_count",
                "team_flag_dispossessions",
                "team_steps_defending_zone",
                "team_steps_attacking_zone",
                "team_steps_adj_teammate",
                "team_steps_adj_opponent",
                "team_blocks_laid",
                "team_blocks_mined",
                "team_blocks_laid_distance_from_own_flag",
                "team_blocks_laid_distance_from_opp_flag"
    ]

    agent_metrics = [
                "agent_flag_pickups",
                "agent_flag_captures",
                "agent_tag_count",
                "agent_respawn_tag_count",
                "agent_flag_dispossessions",
                "agent_steps_defending_zone",
                "agent_steps_attacking_zone",
                "agent_steps_adj_teammate",
                "agent_steps_adj_opponent",
                "agent_blocks_laid",
                "agent_blocks_mined",
                "agent_blocks_laid_distance_from_own_flag",
                "agent_blocks_laid_distance_from_opp_flag",
    ]

    plt_pos = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2),
            9: (3, 0),
            10: (3, 1),
            11: (3, 2),
            12: (4, 0)
    }

    agent_type_labels = {
        0: 'Scout',
        1: 'Guardian',
        2: 'Vaulter',
        3: 'Miner'
    }

    metric_labels2 = {
        0: "flag pickups",
        1: "flag captures",
        2: "tag count",
        3: "respawn tag count",
        4: "flag dispossessions",
        5: "steps defending zone",
        6: "steps attacking zone",
        7: "steps adj. teammate",
        8: "steps adj. opponent",
        9: "blocks laid",
        10: "blocks mined",
        11: "blocks laid dist. own flag",
        12: "blocks laid dist. opp flag"
    }

    team_colours = {
        0: 'dodgerblue',
        1: 'red'
    }

    agent_colours = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf',
    }
   
    # Plot agents metrics
    # fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    # for i, metric in enumerate(team_metrics):
    #     ax[plt_pos[i]].set_title(metric_labels2[i])
    #     for agent in league_trainer.metlog.metrics.keys():
    #         ax[plt_pos[i]].plot(league_trainer.metlog.metrics[agent][metric], label=agent)
    #     ax[plt_pos[i]].set_ylabel(metric_labels2[i])
    #     ax[plt_pos[i]].grid(linestyle='dashed', color='lightgrey')
    #     if i==0:
    #         ax[plt_pos[i]].legend()
    # fig.tight_layout()
    # plt.show()

    team_idxs = ['0', '1'] if not time_capsules[0].symmetric_teams else ['0']
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=FIGSIZE)
    for i, metric in enumerate(team_metrics[:N_METRICS]):
        ax[plt_pos[i]].set_title(metric_labels2[i])
        for team_idx in team_idxs:
            temp_list = []
            for time_capsule in time_capsules:
                for agent in time_capsule.metlog.metrics.keys():
                    if agent[1] == team_idx:
                        temp_list.append(time_capsule.metlog.metrics[agent][metric])
            mean = np.mean(temp_list, axis=0)
            std = np.std(temp_list, axis=0)
            upper = mean + std
            lower = mean - std
            ax[plt_pos[i]].plot(mean, 
                                label='Team ' + str(int(team_idx) + 1),
                                color=team_colours[int(team_idx)])
            ax[plt_pos[i]].fill_between(np.arange(len(mean)), 
                                        lower, 
                                        upper,
                                        alpha=0.1, 
                                        facecolor=team_colours[int(team_idx)])
        ax[plt_pos[i]].set_ylabel(metric_labels2[i])
        ax[plt_pos[i]].grid(linestyle='dashed', alpha=0.5)
        ax[plt_pos[i]].xaxis.set_major_locator(ticker.MultipleLocator(10))
        if i==0:
            fig.legend(bbox_to_anchor=[0.1, -0.01], 
                       loc='upper left', 
                       ncol=5,
                       title='legend')
    if include_block_tile_metrics:
        ax[(4, 1)].set_visible(False)
        ax[(4, 2)].set_visible(False)
    fig.tight_layout()
    plt.show()

    # Plot agent metrics
    agent_types_env = time_capsules[0].agent_types
    agent_teams = time_capsules[0].agent_teams
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=FIGSIZE)
    for i, metric in enumerate(agent_metrics[:N_METRICS]):
        ax[plt_pos[i]].set_title(metric_labels2[i])
        for agent_indiv_idx in time_capsules[0].metlog.agent_indiv_idxs:
            temp_list = []
            for time_capsule in time_capsules:
                for agent_idx in time_capsule.metlog.metrics.keys():
                    # Hack to fix bug in metrics logger -> metrics for all agents are logged for both teams, so need to de-dupe
                    if int(agent_idx[1]) == agent_teams[agent_indiv_idx]:
                        temp_list.append(time_capsule.metlog.metrics[agent_idx][metric][agent_indiv_idx])
                agent_type_label = f'Team {agent_teams[agent_indiv_idx] + 1}: '
                agent_type_label += agent_type_labels[agent_types_env[agent_indiv_idx]]

            mean = np.mean(temp_list, axis=0)
            std = np.std(temp_list, axis=0)
            upper = mean + std
            lower = mean - std
            ax[plt_pos[i]].plot(mean, 
                                label=agent_type_label,
                                color=agent_colours[agent_indiv_idx])
            ax[plt_pos[i]].fill_between(np.arange(len(mean)), 
                                        lower, 
                                        upper,
                                        alpha=0.1, 
                                        facecolor=agent_colours[agent_indiv_idx])
        ax[plt_pos[i]].set_ylabel(metric_labels2[i])
        ax[plt_pos[i]].grid(linestyle='dashed', alpha=0.5)
        ax[plt_pos[i]].xaxis.set_major_locator(ticker.MultipleLocator(10))
        if i==0:
            fig.legend(bbox_to_anchor=[0.1, -0.01], 
                       loc='upper left', 
                       ncol=5,
                       title='legend')
    if include_block_tile_metrics:
        ax[(4, 1)].set_visible(False)
        ax[(4, 2)].set_visible(False)
    fig.tight_layout()
    plt.show()

@ray.remote
def ray_duel(env, agent, opponent, idxs, return_result, device):
    return duel(env, agent, opponent, idxs, return_result, device)

def plot_heatmaps(env, agent1, agent2, n_duels=100):

    ray.shutdown() 
    ray.init()

    async_results = []
    for i in range(n_duels):
        async_result = ray_duel.remote(env, agent1, agent2, (0, 1), return_result=False, device='cpu')
        async_results.append(async_result)

    duel_results = ray.get(async_results)

    for i, d in enumerate(duel_results):
        if i == 0:
            visitation_map = copy.deepcopy(d[2]['agent_visitation_maps'])
        else:
            for k in visitation_map.keys():
                visitation_map[k] += d[2]['agent_visitation_maps'][k]

    agent_type_labels = {
        0: 'Scout',
        1: 'Guardian',
        2: 'Vaulter',
        3: 'Miner'
    }

    agent_types_env = env.AGENT_TYPES
    agent_teams = dict(sorted(env.AGENT_TEAMS.items(), key=lambda item: item[1]))

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 2.8))
    for i, agent_indiv_idx in enumerate(agent_teams.keys()):
        plt_title = f'Team {agent_teams[agent_indiv_idx]+1}: '
        plt_title += agent_type_labels[agent_types_env[agent_indiv_idx]]

        ax[i].set_title(plt_title)
        vmap = visitation_map[agent_indiv_idx]
        sns.heatmap(vmap, annot=False, cmap='magma', ax=ax[i], cbar=False)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()

    return visitation_map

def create_gif(env, 
                agent, 
                opponent, 
                device='cpu', 
                max_frames=256, 
                gif_folder='/gifs/',
                gif_name='output.gif',
                delete_temp_files=True,
                duration=200):
    """
    Duelling algorithm.
    """

    step_count = 0
    done = False
    env.reset()

    # Pre-allocate memory for grid_state and metadata_state tensors
    use_action_mask = torch.empty((env.N_AGENTS, 1), dtype=torch.float32, device=device)
    grid_state = torch.empty((env.N_AGENTS, *env.standardise_state(0).shape), dtype=torch.float32, device=device)
    metadata_state = torch.empty((env.N_AGENTS, *env.get_env_metadata(0).shape), dtype=torch.float32, device=device)

    IMAGE_FOLDER_PATH = os.getcwd() + gif_folder
    if not os.path.exists(IMAGE_FOLDER_PATH):
        os.makedirs(IMAGE_FOLDER_PATH)

    image_frames_paths = []

    with torch.no_grad():
        while not done:

            # Render image
            frame_path = IMAGE_FOLDER_PATH + f'tmp_{step_count}.png'
            env.render_image(frame_path=frame_path)
            image_frames_paths.append(frame_path)

            step_count += 1
            actions = []

            for agent_idx in np.arange(env.N_AGENTS):
                # Get global and local states
                use_action_mask[agent_idx] = torch.tensor(
                    env.AGENT_TYPE_ACTION_MASK[env.AGENT_TYPES[agent_idx]], 
                    dtype=torch.float32,
                    device=device)
                grid_state[agent_idx] = torch.tensor(
                    env.standardise_state(agent_idx, reverse_grid=(env.AGENT_TEAMS[agent_idx] != 0)),
                    dtype=torch.float32,
                    device=device,
                )
                metadata_state[agent_idx] = torch.tensor(
                    env.get_env_metadata(agent_idx),
                    dtype=torch.float32,
                    device=device,
                )

                if env.AGENT_TEAMS[agent_idx] == 0:
                    action = agent.get_action(grid_state[agent_idx], metadata_state[agent_idx], use_action_mask[agent_idx])
                else:
                    action = opponent.get_action(grid_state[agent_idx], metadata_state[agent_idx], use_action_mask[agent_idx])
                    action = env.get_reversed_action(action)

                actions.append(action)

            _, _, done = env.step(actions)

            if step_count >= max_frames:
                done = True

    # Create a GIF from the image frames
    gif_path = os.getcwd() + gif_folder + gif_name
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for frame_path in image_frames_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # Delete the image frames
    if delete_temp_files:
        for frame_path in image_frames_paths:
            os.remove(frame_path)

def duel(env, 
         agent, 
         opponent, 
         idxs, 
         return_result=False, 
         device='cpu', 
         max_steps=256, 
         render=False,
         sleep_time=0.01):
    """
    Duelling algorithm.
    """

    step_count = 0
    done = False
    env.reset()

    # Pre-allocate memory for grid_state and metadata_state tensors
    use_action_mask = torch.empty((env.N_AGENTS, 1), dtype=torch.float32, device=device)
    grid_state = torch.empty((env.N_AGENTS, *env.standardise_state(0).shape), dtype=torch.float32, device=device)
    metadata_state = torch.empty((env.N_AGENTS, *env.get_env_metadata(0).shape), dtype=torch.float32, device=device)

    with torch.no_grad():
        while not done:
            step_count += 1

            actions = []

            for agent_idx in np.arange(env.N_AGENTS):
                # Get global and local states
                use_action_mask[agent_idx] = torch.tensor(
                    env.AGENT_TYPE_ACTION_MASK[env.AGENT_TYPES[agent_idx]], 
                    dtype=torch.float32,
                    device=device)
                grid_state[agent_idx] = torch.tensor(
                    env.standardise_state(agent_idx, reverse_grid=(env.AGENT_TEAMS[agent_idx] != 0)),
                    dtype=torch.float32,
                    device=device,
                )
                metadata_state[agent_idx] = torch.tensor(
                    env.get_env_metadata(agent_idx),
                    dtype=torch.float32,
                    device=device,
                )

                if env.AGENT_TEAMS[agent_idx] == 0:
                    action = agent.get_action(grid_state[agent_idx], metadata_state[agent_idx], use_action_mask[agent_idx])
                else:
                    action = opponent.get_action(grid_state[agent_idx], metadata_state[agent_idx], use_action_mask[agent_idx])
                    action = env.get_reversed_action(action)

                actions.append(action)

            _, _, done = env.step(actions)

            if render:
                env.render(sleep_time=sleep_time)
                print(step_count, env.metrics['team_flag_captures'][0], env.metrics['team_flag_captures'][1])

            if step_count > max_steps:
                done = True

    if return_result:
        if env.metrics['team_flag_captures'][0] > env.metrics['team_flag_captures'][1]:
            result = 1
        elif env.metrics['team_flag_captures'][0] == env.metrics['team_flag_captures'][1]:
            result = 0
        else:
            result = - 1
        return_metrics = result
    else:
        return_metrics = env.metrics
    
    return idxs[0], idxs[1], return_metrics

def winrate_heatmap(results_df):
    """
    Create winrate heatmap.
    """
    plt.figure()
    plt.title('League Training Win Rates')
    # sns.heatmap(results_df, annot=True, annot_kws={'size': 8}, cmap='magma')
    sns.heatmap(results_df, annot=False, cmap='rocket')
    plt.yticks(rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.show()


def create_results_table(run_dir, file_names, idx=-1, equal_var=False):
    """
    Create results table.
    """
    
    team_metrics = [
                "team_flag_pickups",
                "team_flag_captures",
                "team_tag_count",
                "team_respawn_tag_count",
                "team_flag_dispossessions",
                "team_steps_defending_zone",
                "team_steps_attacking_zone",
                "team_steps_adj_teammate",
                "team_steps_adj_opponent",
                "team_blocks_laid",
                "team_blocks_mined",
                "team_blocks_laid_distance_from_own_flag",
                "team_blocks_laid_distance_from_opp_flag"
    ]

    agent_metrics = [
                    "agent_flag_pickups",
                    "agent_flag_captures",
                    "agent_tag_count",
                    "agent_respawn_tag_count",
                    "agent_flag_dispossessions",
                    "agent_steps_defending_zone",
                    "agent_steps_attacking_zone",
                    "agent_steps_adj_teammate",
                    "agent_steps_adj_opponent",
                    "agent_blocks_laid",
                    "agent_blocks_mined",
                    "agent_blocks_laid_distance_from_own_flag",
                    "agent_blocks_laid_distance_from_opp_flag",
    ]

    # Load data from time capsules
    time_capsules = []
    for file_name in file_names:
        file_path = run_dir + file_name
        print(file_path)
        
        with open(file_path, 'rb') as f:
            time_capsules.append(pickle.load(f))

    # Team metrics
    team_results = defaultdict(list)
    team_idxs = ['0', '1'] if not time_capsules[0].symmetric_teams else ['0']
    for i, metric in enumerate(team_metrics):
        metric_samples = {}
        for team_idx in team_idxs:
            temp_list = []
            for time_capsule in time_capsules:
                for agent in time_capsule.metlog.metrics.keys():
                    if agent[1] == team_idx:
                        temp_list.append(time_capsule.metlog.metrics[agent][metric][idx])
            metric_samples[team_idx] = temp_list

        _, p_value_teams = stats.ttest_ind(metric_samples['0'], metric_samples['1'], equal_var=equal_var)   
        team1_mean = np.mean(metric_samples['0'], axis=0)
        team2_mean = np.mean(metric_samples['1'], axis=0)
        team1_std = np.std(metric_samples['0'], axis=0)
        team2_std = np.std(metric_samples['1'], axis=0)  

        team_results['metric_name'].append(metric)
        team_results['team1_mean'].append(round(team1_mean, 4))
        team_results['team1_std'].append(round(team1_std, 4))
        team_results['team2_mean'].append(round(team2_mean, 4))
        team_results['team2_std'].append(round(team2_std, 4))
        team_results['p_value'].append(round(p_value_teams, 4))
        
    # Agent metrics
    agent_results_t1 = defaultdict(list)
    agent_results_t2 = defaultdict(list)
    agent_types_env = time_capsules[0].agent_types
    agent_teams = time_capsules[0].agent_teams
    for i, metric in enumerate(agent_metrics):
        agent_samples = {}
        for agent_indiv_idx in time_capsules[0].metlog.agent_indiv_idxs:
            temp_list = []
            for time_capsule in time_capsules:
                for agent_idx in time_capsule.metlog.metrics.keys():
                    # Hack to fix bug in metrics logger -> metrics for all agents are logged for both teams, so need to de-dupe
                    if int(agent_idx[1]) == agent_teams[agent_indiv_idx]:
                        temp_list.append(time_capsule.metlog.metrics[agent_idx][metric][agent_indiv_idx][idx])
            agent_samples[agent_indiv_idx] = temp_list

        _, p_value_02 = stats.ttest_ind(agent_samples[0], agent_samples[2], equal_var=equal_var)
        _, p_value_13 = stats.ttest_ind(agent_samples[1], agent_samples[3], equal_var=equal_var)
        agent0_mean = np.mean(agent_samples[0], axis=0)
        agent2_mean = np.mean(agent_samples[2], axis=0)
        agent0_std = np.std(agent_samples[0], axis=0)
        agent2_std = np.std(agent_samples[2], axis=0)

        agent_results_t1['metric_name'].append(metric)
        agent_results_t1['agent1_mean'].append(round(agent0_mean, 2))
        agent_results_t1['agent1_std'].append(round(agent0_std, 2))
        agent_results_t1['agent2_mean'].append(round(agent2_mean, 2))
        agent_results_t1['agent2_std'].append(round(agent2_std, 2))
        agent_results_t1['p_value'].append(round(p_value_02, 4))
        
        agent1_mean = np.mean(agent_samples[1], axis=0)
        agent3_mean = np.mean(agent_samples[3], axis=0)
        agent1_std = np.std(agent_samples[1], axis=0)
        agent3_std = np.std(agent_samples[3], axis=0)

        agent_results_t2['metric_name'].append(metric)
        agent_results_t2['agent1_mean'].append(round(agent1_mean, 2))
        agent_results_t2['agent1_std'].append(round(agent1_std, 2))
        agent_results_t2['agent2_mean'].append(round(agent3_mean, 2))
        agent_results_t2['agent2_std'].append(round(agent3_std, 2))
        agent_results_t2['p_value'].append(round(p_value_13, 4))

    team_results_df = pd.DataFrame(team_results)
    agent_results_t1_df = pd.DataFrame(agent_results_t1)
    agent_results_t2_df = pd.DataFrame(agent_results_t2)

    return team_results_df, agent_results_t1_df, agent_results_t2_df