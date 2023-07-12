# Plot rewards
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 
import torch
import os
import pickle
import imageio


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
                file_names):
    """
    Create primary plots for analysis.
    """
    
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
                "agent_blocks_laid",
                "agent_blocks_mined",
                "agent_blocks_laid_distance_from_own_flag",
                "agent_blocks_laid_distance_from_opp_flag"
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
            10: (3, 1)
    }

    agent_type_labels = {
        0: 'Scout',
        1: 'Guardian',
        2: 'Vaulter',
        3: 'Miner'
    }

    metric_labels1 = {
        0: 'Team ',
        1: 'Team ',
        2: 'Agent '
    }

    metric_labels2 = {
        0: "flag pickups",
        1: "flag captures",
        2: "tag count",
        3: "respawn tag count",
        4: "flag dispossessions",
        5: "steps defending zone",
        6: "steps attacking zone",
        7: "blocks laid",
        8: "blocks mined",
        9: "blocks laid dist. own flag",
        10: "blocks laid dist. opp flag"
    }

    team_colours = {
        0: 'dodgerblue',
        1: 'red'
    }

    agent_colours = {
        0: '#f4a460',
        1: '#228b22',
        2: '#9370db',
        3: '#ff00ff',
        4: '#6495ed',
        5: '#ffff00',
        6: '#006400',
        7: '#00008b',
        8: '#b03060',
        9: '#ff0000',
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
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for i, metric in enumerate(team_metrics):
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
        if i==0:
            # ax[plt_pos[i]].legend()
            fig.legend(bbox_to_anchor=[0.1, -0.01], 
                       loc='upper left', 
                       ncol=5,
                       title='legend')
        if i == 11:
            ax[plt_pos[i]].set_visible(False)
    fig.tight_layout()
    plt.show()

    # Plot agent metrics
    agent_types_env = time_capsules[0].agent_types
    agent_teams = time_capsules[0].agent_teams
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for i, metric in enumerate(agent_metrics):
        ax[plt_pos[i]].set_title(metric_labels2[i])
        for agent_indiv_idx in time_capsules[0].metlog.agent_indiv_idxs:
            temp_list = []
            for time_capsule in time_capsules:
                for agent_idx in time_capsule.metlog.metrics.keys():
                    temp_list.append(time_capsule.metlog.metrics[agent_idx][metric][agent_indiv_idx])
                agent_type_label = f'Team {agent_teams[agent_indiv_idx] + 1}: '
                agent_type_label += agent_type_labels[agent_types_env[agent_indiv_idx]]
            # ax[plt_pos[i]].plot(np.mean(temp_list, axis=0), label=agent_type_label, color=agent_colours[agent_indiv_idx])
            # ax[plt_pos[i]].plot(np.mean(temp_list, axis=0), label=agent_type_label)

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
        if i==0:
            # ax[plt_pos[i]].legend()
            fig.legend(bbox_to_anchor=[0.1, -0.01], 
                       loc='upper left', 
                       ncol=5,
                       title='legend')
        if i == 11:
            ax[plt_pos[i]].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_heatmaps(env):
    """
    Plot agent heatmaps.
    """
    plt_idxs = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }

    agent_type_labels = {
        0: 'Scout',
        1: 'Guardian',
        2: 'Vaulter',
        3: 'Miner'
    }

    agent_types_env = env.AGENT_TYPES
    agent_teams = env.AGENT_TEAMS

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    for agent_indiv_idx in range(4):
        plt_title = f'Team {agent_teams[agent_indiv_idx]+1}: '
        plt_title += agent_type_labels[agent_types_env[agent_indiv_idx]]

        ax[plt_idxs[agent_indiv_idx]].set_title(plt_title)
        vmap = env.metrics['agent_visitation_maps'][agent_indiv_idx]
        sns.heatmap(vmap, annot=False, cmap='magma', ax=ax[plt_idxs[agent_indiv_idx]], cbar=False)
        ax[plt_idxs[agent_indiv_idx]].get_xaxis().set_visible(False)
        ax[plt_idxs[agent_indiv_idx]].get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()

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