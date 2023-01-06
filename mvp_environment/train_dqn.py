import torch
import numpy as np
from IPython.display import clear_output
import utils as ut
from collections import defaultdict

def train_dqn(env,
            agent_t1,
            agent_t2,
            env_dims,
            epochs,
            batch_size,
            n_agents=4,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.1,
            n_random_steps=0,
            max_steps=1000,
            learning_skip_steps=1,
            use_standardised_state=True,
            device='cpu'):
    """
    Train DQN agent.
    """

    step_count = 0
    done_count = 0

    training_metrics = {
        "score_history": [], 
        "losses": [],
        "team_1_captures": [], 
        "team_2_captures": [],
        "team_1_tags": [],
        "team_2_tags": [],
        "episode_step_counts": [],
        "agent_tag_count": defaultdict(list),
        "agent_flag_captures": defaultdict(list),
        "agent_blocks_laid": defaultdict(list),
        "agent_blocks_mined": defaultdict(list),
        "agent_avg_distance_to_own_flag": defaultdict(list),
        "agent_avg_distance_to_opp_flag": defaultdict(list),
        "agent_health_pickups": defaultdict(list),
    }

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

            # Collect actions for each agent
            actions =[]
            for agent_idx in np.arange(n_agents):
                #curr_metadata_state = ut.get_env_metadata(agent_idx, env.has_flag, env.agent_types_np, device=device)
                curr_metadata_state = torch.from_numpy(env.get_env_metadata(agent_idx)).reshape(1, 22).float().to(device)
                if use_standardised_state:
                    curr_grid_state_ = env.get_standardised_state(agent_idx) + ut.add_noise(env_dims)
                    curr_grid_state = torch.from_numpy(curr_grid_state_).float().to(device)
                if env.AGENT_TEAMS[agent_idx]==0:
                    actions.append(agent_t1.choose_action(curr_grid_state, curr_metadata_state))
                else:
                    actions.append(agent_t2.choose_action(curr_grid_state, curr_metadata_state))

            # Step the environment
            new_grid_state, rewards, done = env.step(actions)
            new_grid_state_ = new_grid_state.reshape(*env_dims) + ut.add_noise(env_dims)
            new_grid_state = torch.from_numpy(new_grid_state_).float().to(device)

            # Increment score
            score += sum(rewards)

            # Store each agent experiences
            for agent_idx in np.arange(n_agents):
                #new_metadata_state = ut.get_env_metadata(agent_idx, env.has_flag, env.agent_types_np, device=device)
                new_metadata_state = torch.from_numpy(env.get_env_metadata(agent_idx)).reshape(1, 22).float().to(device)

                # Append replay buffer
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
            
            # Learning
            if step_count>n_random_steps \
               and step_count%learning_skip_steps==0 \
               and min(len(agent_t1.memory), len(agent_t2.memory)) > batch_size:
                loss_t1 = agent_t1.update_network(step_count)
                loss_t2 = agent_t2.update_network(step_count)
            else:
                loss_t1 = 0.0
                loss_t2 = 0.0

            # Losses
            training_metrics['losses'].append((loss_t1, loss_t2))   

            # Termination -> Append metrics
            if done or episode_step_count > max_steps:
                training_metrics['team_1_captures'].append(env.metrics['team_points'][0])
                training_metrics['team_2_captures'].append(env.metrics['team_points'][1])
                training_metrics['team_1_tags'].append(env.metrics['tag_count'][0])
                training_metrics['team_2_tags'].append(env.metrics['tag_count'][1])
                for agent_idx in range(n_agents):
                    training_metrics["agent_tag_count"][agent_idx].append(env.metrics['agent_tag_count'][agent_idx])
                    training_metrics["agent_flag_captures"][agent_idx].append(env.metrics['agent_flag_captures'][agent_idx])
                    training_metrics["agent_blocks_laid"][agent_idx].append(env.metrics['agent_blocks_laid'][agent_idx])
                    training_metrics["agent_blocks_mined"][agent_idx].append(env.metrics['agent_blocks_mined'][agent_idx])
                    training_metrics["agent_avg_distance_to_own_flag"][agent_idx].append(env.metrics['agent_total_distance_to_own_flag'][agent_idx]/env.env_step_count)
                    training_metrics["agent_avg_distance_to_opp_flag"][agent_idx].append(env.metrics['agent_total_distance_to_opp_flag'][agent_idx]/env.env_step_count)
                    training_metrics["agent_health_pickups"][agent_idx].append(env.metrics['agent_health_pickups'][agent_idx])

                done_count += 1 * done
                done = True
                
        # Decay epsilon
        epsilon_ = max(epsilon * epsilon_decay**i, epsilon_min)
        agent_t1.epsilon = epsilon_
        agent_t2.epsilon = epsilon_

        training_metrics['score_history'].append(score)
        training_metrics['episode_step_counts'].append(episode_step_count)

        if i % 5 == 0:
            clear_output(wait=True)
            print(f"episode: {i+1} \
                \ntotal step count: {step_count} \
                \nepisode step count: {episode_step_count} \
                \nscore: {score} \
                \naverage score: {np.mean(training_metrics['score_history'][-100:])} \
                \nepsilon: {round(epsilon_, 4)} \
                \ndone count: {done_count} \
                \nteam 1 captures: {sum(training_metrics['team_1_captures'])} \
                \nteam 2 captures: {sum(training_metrics['team_2_captures'])}" )


    return training_metrics