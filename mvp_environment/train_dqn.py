import torch
import numpy as np
from IPython.display import clear_output
import utils as ut


def train_dqn(env,
            agent_t1,
            agent_t2,
            env_dims,
            epochs,
            batch_size,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.1,
            n_random_steps=0,
            max_steps=1000,
            learning_skip_steps = 1,
            device='cpu'):
    """
    Train DQN agent.
    """

    step_count = 0
    done_count = 0
    team_1_captures = []
    team_2_captures = []
    episode_step_counts = []
    losses = []
    score_history = []

    for i in range(epochs):
        env.reset()

        curr_grid_state_ = env.grid.reshape(*env_dims) + ut.add_noise(env_dims)
        curr_grid_state = torch.from_numpy(curr_grid_state_).float().to(device)

        done = False
        episode_step_count = 0
        score = 0
        while not done: 
            step_count += 1
            episode_step_count += 1

            # collect actions for each agent
            actions =[]
            for agent_idx in np.arange(4):
                curr_metadata_state = ut.get_env_metadata(agent_idx, env.has_flag, device=device)
                if env.AGENT_TEAMS[agent_idx]==0:
                    actions.append(agent_t1.choose_action(curr_grid_state, curr_metadata_state))
                else:
                    actions.append(agent_t2.choose_action(curr_grid_state, curr_metadata_state))

            # step the environment
            new_grid_state, rewards, done = env.step(actions)
            new_grid_state_ = new_grid_state.reshape(*env_dims) + ut.add_noise(env_dims)
            new_grid_state = torch.from_numpy(new_grid_state_).float().to(device)

            # increment score
            score += sum(rewards)

            # store each agent experiences
            for agent_idx in np.arange(4):
                new_metadata_state = ut.get_env_metadata(agent_idx, env.has_flag, device=device)

                # append replay buffer
                if env.AGENT_TEAMS[agent_idx]==0:
                    agent_t1.memory.append(
                                        ((curr_grid_state, curr_metadata_state), 
                                        actions[agent_idx], 
                                        rewards[agent_idx], 
                                        (new_grid_state, new_metadata_state), 
                                        done)
                                        )
                else:
                    agent_t2.memory.append(
                                        ((curr_grid_state, curr_metadata_state), 
                                        actions[agent_idx], 
                                        rewards[agent_idx], 
                                        (new_grid_state, new_metadata_state), 
                                        done)
                                        )

            curr_grid_state = new_grid_state
            
            # learning
            if step_count>n_random_steps \
               and step_count%learning_skip_steps==0 \
               and min(len(agent_t1.memory), len(agent_t2.memory)) > batch_size:
                loss_t1 = agent_t1.update_network(step_count)
                loss_t2 = agent_t2.update_network(step_count)
            else:
                loss_t1 = 0.0
                loss_t2 = 0.0

            # append metrics
            losses.append((loss_t1, loss_t2))  

            # termination
            if done or episode_step_count > max_steps:
                team_1_captures.append(env.team_points[0])
                team_2_captures.append(env.team_points[1])
                done_count += 1 * done
                done = True
                
        # decay epsilon
        epsilon_ = max(epsilon * epsilon_decay**i, epsilon_min)
        agent_t1.epsilon = epsilon_
        agent_t2.epsilon = epsilon_

        score_history.append(score)
        episode_step_counts.append(episode_step_count)

        if i % 5 == 0:
            clear_output(wait=True)
            print(f'episode: {i} \ntotal step count: {step_count} \nepisode step count: {episode_step_count} \
                \nscore: {score} \naverage score: {np.mean(score_history[-100:])} \
                \nepsilon: {round(epsilon_, 4)} \ndone count: {done_count} \
                \nteam 1 captures: {sum(team_1_captures)} \nteam 2 captures: {sum(team_2_captures)}')

    return score_history, losses, team_1_captures, team_2_captures, episode_step_counts