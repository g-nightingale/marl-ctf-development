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

def test_model(env, agents, env_dims, display=True, max_moves=50, device='cpu'):
    """
    Test the trained agent policies.
    """
    env.reset()

    grid_state_ = env.grid.reshape(*env_dims) + ut.add_noise(env_dims)
    grid_state = torch.from_numpy(grid_state_).float().to(device)

    done = False
    step_count = 0
    score = 0
    while not done: 
        step_count += 1

        # Collect actions for each agent
        actions =[]
        for agent_idx in np.arange(4):
            metadata_state = ut.get_env_metadata(agent_idx, env.has_flag, env.agent_types_np)
            actions.append(agents[agent_idx].choose_action(grid_state, metadata_state))

        # Step the environment
        grid_state, rewards, done = env.step(actions)
        grid_state_ = grid_state.reshape(*env_dims) + ut.add_noise(env_dims)
        grid_state = torch.from_numpy(grid_state_).float().to(device)

        # Increment score
        score += sum(rewards)

        if display:
            env.render(sleep_time=0.1)

        if done:
            if display:
                print(f"Game won! \
                      \nFinal score: {score} \
                      \nTeam 1 score: {env.team_points[0]} \
                      \nTeam 2 score: {env.team_points[1]} \
                      \nTotal moves: {step_count}")
       
        if (step_count > max_moves):
            if display:
                print(f"Move limit reached. \
                      \nFinal score: {score} \
                      \nTeam 1 score: {env.team_points[0]} \
                      \nTeam 2 score: {env.team_points[1]} \
                      \nTotal moves: {step_count}")
            break

def plot_training_performance(score_history, 
                              losses, 
                              team_1_captures, 
                              team_2_captures, 
                              team_1_tags,
                              team_2_tags,
                              episode_step_counts):
    """
    Plot training performance metrics.
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

    ax[0, 0].set_title('sum of rewards')
    ax[0, 0].plot(score_history, label='episode score', c='darkorange')
    ax[0, 0].plot([np.mean(score_history[::-1][i:i+100]) for i in range(len(score_history))][::-1], label='average reward (last 100 episodes)', c='green')
    ax[0, 0].set_xlabel("episodes")
    ax[0, 0].set_ylabel("reward")
    ax[0, 0].legend()

    ax[0, 1].set_title('losses')
    ax[0, 1].plot([l[0] for l in losses], label='team 1 losses', c='mediumblue')
    ax[0, 1].plot([l[1] for l in losses], label='team 2 losses', c='firebrick')
    ax[0, 1].set_xlabel("steps")
    ax[0, 1].set_ylabel("losses")
    ax[0, 1].legend()

    ax[1, 0].set_title('team cumulative flag captures')
    ax[1, 0].plot(np.cumsum(team_1_captures), label='team 1 cumulative flag captures', c='mediumblue')
    ax[1, 0].plot(np.cumsum(team_2_captures), label='team 2 cumulative flag captures', c='firebrick')
    ax[1, 0].set_xlabel("episodes")
    ax[1, 0].set_ylabel("cumulative wins")
    ax[1, 0].legend()

    ax[1, 1].set_title('team cumulative tags')
    ax[1, 1].plot(np.cumsum(team_1_tags), label='team 1 cumulative tags', c='mediumblue')
    ax[1, 1].plot(np.cumsum(team_2_tags), label='team 2 cumulative tags', c='firebrick')
    ax[1, 1].set_xlabel("episodes")
    ax[1, 1].set_ylabel("cumulative tags")
    ax[1, 1].legend()

    ax[2, 0].set_title('episode duration')
    ax[2, 0].plot(episode_step_counts, label='episode duration', c='darkorange')
    ax[2, 0].plot([np.mean(episode_step_counts[::-1][i:i+100]) for i in range(len(score_history))][::-1], label='average episode duration (last 100 episodes)', c='green')
    ax[2, 0].set_xlabel("episodes")
    ax[2, 0].set_ylabel("episode duration (steps)")
    ax[2, 0].legend()

    ax[2, 1].set_axis_off()

    fig.tight_layout()
    
    plt.show()
