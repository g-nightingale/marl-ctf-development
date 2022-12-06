import numpy as np
import torch
import utils as ut


def add_noise(env_dims):
    """
    Add noise to stabilise training.
    """
    return np.random.rand(*env_dims)/100.0

def get_env_metadata(agent_idx, has_flag, device='cpu'):
    """
    Get agent turn and flag holder info.
    """
    agent_turn = np.array([0, 0, 0, 0], dtype=np.int8)
    agent_turn[agent_idx] = 1

    return torch.cat((torch.from_numpy(agent_turn), torch.from_numpy(has_flag))).reshape(1, 8).float().to(device)

def preprocess(grid, max_value=8.0):
    """
    Preprocess the grid to fall between 0 and 1.
    """
    return np.divide(grid, max_value, dtype=np.float16)

def test_model(env, agent_t1, agent_t2, env_dims, display=True, max_moves=50, device='cpu'):
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
            metadata_state = ut.get_env_metadata(agent_idx, env.has_flag)
            if env.AGENT_TEAMS[agent_idx]==0:
                actions.append(agent_t1.choose_action(grid_state, metadata_state))
            else:
                actions.append(agent_t2.choose_action(grid_state, metadata_state))

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

