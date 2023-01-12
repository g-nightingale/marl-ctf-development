import numpy as np
import torch
import utils as ut
import matplotlib.pyplot as plt


def add_noise(env_dims):
    """
    Add noise to stabilise training.
    """
    return np.random.rand(*env_dims)/100.0

def get_env_metadata(agent_idx, has_flag, agent_types, device='cpu'):
    """
    Get agent turn and flag holder info.
    """
    agent_turn = np.array([0, 0, 0, 0], dtype=np.int8)
    agent_turn[agent_idx] = 1

    return torch.cat((torch.from_numpy(agent_turn), torch.from_numpy(agent_types), \
                      torch.from_numpy(has_flag))).reshape(1, 12).float().to(device)

def preprocess(grid, max_value=8.0):
    """
    Preprocess the grid to fall between 0 and 1.
    """
    return np.divide(grid, max_value, dtype=np.float16)

def test_model_ppo(env, 
                agent_t1, 
                agent_t2, 
                display=True, 
                use_ego_state=False,
                scale_tiles=False,
                max_moves=50, 
                device='cpu'):
    """
    Test the trained agent policies.
    """
    env.reset()

    done = False
    step_count = 0
    score = 0

    if use_ego_state:
        env_dims = env.EGO_ENV_DIMS
    else:
        env_dims = env.ENV_DIMS

    while not done: 
        step_count += 1

        # Collect actions for each agent
        actions =[]
        for agent_idx in np.arange(env.N_AGENTS):
            metadata_state = torch.from_numpy(env.get_env_metadata(agent_idx)).reshape(1, env.METADATA_VECTOR_LEN).float().to(device)
            
            grid_state_ = env.standardise_state(agent_idx, use_ego_state=use_ego_state, scale_tiles=scale_tiles).reshape(*env_dims) + ut.add_noise(env_dims)
            grid_state = torch.from_numpy(grid_state_).float().to(device)

            if env.AGENT_TEAMS[agent_idx]==0:
                action, _, _ = agent_t1.choose_action(grid_state, metadata_state)
                actions.append(action)
            else:
                action, _, _ = agent_t2.choose_action(grid_state, metadata_state)
                actions.append(action)

        # Step the environment
        _, rewards, done = env.step(actions)

        # Increment score
        score += sum(rewards)

        if display:
            env.render(sleep_time=0.1)

        if done:
            if display:
                print(f"Game won! \
                      \nFinal score: {score} \
                      \nTeam 1 score: {env.metrics['team_points'][0]} \
                      \nTeam 2 score: {env.metrics['team_points'][1]} \
                      \nTotal moves: {step_count}")
       
        if (step_count > max_moves):
            if display:
                print(f"Move limit reached. \
                      \nFinal score: {score} \
                      \nTeam 1 score: {env.metrics['team_points'][0]} \
                      \nTeam 2 score: {env.metrics['team_points'][1]} \
                      \nTotal moves: {step_count}")
            break

def test_model_dqn(env, 
                agent_t1, 
                agent_t2, 
                display=True, 
                use_ego_state=False,
                scale_tiles=False,
                max_moves=50, 
                device='cpu'):
    """
    Test the trained agent policies.
    """
    env.reset()

    done = False
    step_count = 0
    score = 0

    if use_ego_state:
        env_dims = env.EGO_ENV_DIMS
    else:
        env_dims = env.ENV_DIMS

    while not done: 
        step_count += 1

        # Collect actions for each agent
        actions =[]
        for agent_idx in np.arange(env.N_AGENTS):
            metadata_state = torch.from_numpy(env.get_env_metadata(agent_idx)).reshape(1, env.METADATA_VECTOR_LEN).float().to(device)
            
            grid_state_ = env.standardise_state(agent_idx, use_ego_state=use_ego_state, scale_tiles=scale_tiles).reshape(*env_dims) + ut.add_noise(env_dims)
            grid_state = torch.from_numpy(grid_state_).float().to(device)

            if env.AGENT_TEAMS[agent_idx]==0:
                actions.append(agent_t1.choose_action(grid_state, metadata_state))
            else:
                actions.append(agent_t2.choose_action(grid_state, metadata_state))

        # Step the environment
        _, rewards, done = env.step(actions)

        # Increment score
        score += sum(rewards)

        if display:
            env.render(sleep_time=0.1)

        if done:
            if display:
                print(f"Game won! \
                      \nFinal score: {score} \
                      \nTeam 1 score: {env.metrics['team_points'][0]} \
                      \nTeam 2 score: {env.metrics['team_points'][1]} \
                      \nTotal moves: {step_count}")
       
        if (step_count > max_moves):
            if display:
                print(f"Move limit reached. \
                      \nFinal score: {score} \
                      \nTeam 1 score: {env.metrics['team_points'][0]} \
                      \nTeam 2 score: {env.metrics['team_points'][1]} \
                      \nTotal moves: {step_count}")
            break

def plot_training_performance(training_metrics):
    """
    Plot training performance metrics.
    """
    
    AVG_PERIOD = 100
    AGENT_COLOURS = ['mediumblue', 'firebrick', 'mediumblue', 'firebrick']
    AGENT_LINESTYLE = ['solid', 'solid', 'dashed', 'dashed']
    LINEWIDTH = 1.0

    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(10, 18))

    losses_t1 = [l[0] for l in training_metrics['losses']]
    losses_t2 = [l[1] for l in training_metrics['losses']]

    # Rewards & episode duration
    ax[0, 0].set_title('sum of rewards')
    ax[0, 0].plot(training_metrics['score_history'], label='episode score', c='darkorange', linewidth=LINEWIDTH)
    ax[0, 0].plot([np.mean(training_metrics['score_history'][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['score_history']))][::-1], 
                           label=f'average reward (last {AVG_PERIOD} episodes)', c='green')
    ax[0, 0].set_xlabel("episodes")
    ax[0, 0].set_ylabel("reward")
    ax[0, 0].legend()

    ax[0, 1].set_title('episode duration')
    ax[0, 1].plot(training_metrics['episode_step_counts'], label='episode duration', c='darkorange', linewidth=LINEWIDTH)
    ax[0, 1].plot([np.mean(training_metrics['episode_step_counts'][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['score_history']))][::-1], 
                           label=f'average episode duration (last {AVG_PERIOD} episodes)', c='green')
    ax[0, 1].set_xlabel("episodes")
    ax[0, 1].set_ylabel("episode duration (steps)")
    ax[0, 1].legend()

    # Team level metrics
    ax[1, 0].set_title('team cumulative flag captures')
    ax[1, 0].plot(np.cumsum(training_metrics['team_1_captures']), label='team 1 cumulative flag captures', c='mediumblue', linewidth=LINEWIDTH)
    ax[1, 0].plot(np.cumsum(training_metrics['team_2_captures']), label='team 2 cumulative flag captures', c='firebrick', linewidth=LINEWIDTH)
    ax[1, 0].set_xlabel("episodes")
    ax[1, 0].set_ylabel("cumulative wins")
    ax[1, 0].legend()

    ax[1, 1].set_title('team cumulative tags')
    ax[1, 1].plot(np.cumsum(training_metrics['team_1_tags']), label='team 1 cumulative tags', c='mediumblue', linewidth=LINEWIDTH)
    ax[1, 1].plot(np.cumsum(training_metrics['team_2_tags']), label='team 2 cumulative tags', c='firebrick', linewidth=LINEWIDTH)
    ax[1, 1].set_xlabel("episodes")
    ax[1, 1].set_ylabel("cumulative tags")
    ax[1, 1].legend()

    # Agent level metrics
    ax[2, 0].set_title(f'agent flag captures - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        #ax[2, 0].plot(np.cumsum(training_metrics['agent_flag_captures'][i]), c=AGENT_COLOURS[i], linestyle=AGENT_LINESTYLE[i], linewidth=LINEWIDTH, alpha=0.1)
        ax[2, 0].plot([np.mean(training_metrics['agent_flag_captures'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_flag_captures'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[2, 0].set_xlabel("episodes")
    ax[2, 0].set_ylabel("flag captures")
    ax[2, 0].legend()

    ax[2, 1].set_title(f'agent tag count - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        ax[2, 1].plot([np.mean(training_metrics['agent_tag_count'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_tag_count'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[2, 1].set_xlabel("episodes")
    ax[2, 1].set_ylabel("tag count")
    ax[2, 1].legend()

    ax[3, 0].set_title(f'agent blocks laid - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        ax[3, 0].plot([np.mean(training_metrics['agent_blocks_laid'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_blocks_laid'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[3, 0].set_xlabel("episodes")
    ax[3, 0].set_ylabel("blocks laid")
    ax[3, 0].legend()

    ax[3, 1].set_title(f'agent blocks mined - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        ax[3, 1].plot([np.mean(training_metrics['agent_blocks_mined'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_blocks_mined'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[3, 1].set_xlabel("episodes")
    ax[3, 1].set_ylabel("blocks mined")
    ax[3, 1].legend()

    ax[4, 0].set_title(f'average distance from own flag - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        ax[4, 0].plot([np.mean(training_metrics['agent_avg_distance_to_own_flag'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_avg_distance_to_own_flag'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[4, 0].set_xlabel("episodes")
    ax[4, 0].set_ylabel("avg distance")
    ax[4, 0].legend()

    ax[4, 1].set_title(f'average distance from opp flag - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        ax[4, 1].plot([np.mean(training_metrics['agent_avg_distance_to_opp_flag'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_avg_distance_to_opp_flag'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[4, 1].set_xlabel("episodes")
    ax[4, 1].set_ylabel("avg distance")
    ax[4, 1].legend()

    ax[5, 0].set_title(f'total agent health pickups - avg last {AVG_PERIOD} episodes')
    for a in range(4):
        ax[5, 0].plot([np.mean(training_metrics['agent_health_pickups'][a][::-1][i:i+AVG_PERIOD]) for i in range(len(training_metrics['agent_health_pickups'][a]))][::-1], 
                               label=f'agent {a+1}', c=AGENT_COLOURS[a], linestyle=AGENT_LINESTYLE[a], linewidth=LINEWIDTH)
    ax[5, 0].set_xlabel("episodes")
    ax[5, 0].set_ylabel("total health pickups")
    ax[5, 0].legend()


    ax[5, 1].set_title(f'losses - avg last {AVG_PERIOD} episodes')
    ax[5, 1].plot([np.mean(losses_t1[::-1][i:i+AVG_PERIOD]) for i in range(len(losses_t1))][::-1], label='team 1 losses', c='mediumblue', linewidth=LINEWIDTH)
    ax[5, 1].plot([np.mean(losses_t2[::-1][i:i+AVG_PERIOD]) for i in range(len(losses_t2))][::-1], label='team 2 losses', c='firebrick', linewidth=LINEWIDTH)
    ax[5, 1].set_xlabel("episodes")
    ax[5, 1].set_ylabel("losses")
    ax[5, 1].legend()

    # Remove empty axis
    #ax[5, 1].set_axis_off()

    fig.tight_layout()
    
    plt.show()