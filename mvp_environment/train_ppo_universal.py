import numpy as np
from ppo import PPOAgent
from gridworld_ctf_mvp import GridworldCtf
from collections import defaultdict
import torch
from IPython.display import clear_output
import utils as ut

def train_ppo_universal(env,
            uni_agent,
            n_episodes,
            learning_steps=20,
            max_steps=1000,
            use_ego_state=False,
            scale_tiles=False,
            device='cpu'):
    """
    Train PPO agent.
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


    for i in range(n_episodes):
        env.reset()
        done = False
        episode_step_count = 0
        score = 0

        if use_ego_state:
            env_dims = env.EGO_ENV_DIMS
        else:
            env_dims = env.ENV_DIMS

        while not done: 
            step_count += 1
            episode_step_count += 1

            # Collect actions for each agent
            actions = []
            probs = []
            vals = []
            for agent_idx in np.arange(env.N_AGENTS):
                curr_metadata_state = torch.from_numpy(env.get_env_metadata(agent_idx)).reshape(1, env.METADATA_VECTOR_LEN).float().to(device)

                # If using the standardised states, get the agent specific states
                curr_grid_state_ = env.standardise_state(agent_idx, use_ego_state=use_ego_state, scale_tiles=scale_tiles).reshape(*env_dims) + ut.add_noise(env_dims)
                curr_grid_state = torch.from_numpy(curr_grid_state_).float().to(device)

                #curr_grid_state = env.standardise_state(agent_idx, use_ego_state=use_ego_state, scale_tiles=scale_tiles).reshape(*env_dims) + ut.add_noise(env_dims)
                action, prob, val = uni_agent.choose_action(curr_grid_state, curr_metadata_state)
                actions.append(action)
                probs.append(prob)
                vals.append(val)

            # Step the environment
            _, rewards, done = env.step(actions)

            # Increment score
            score += sum(rewards)

            # Store each agent experiences
            for agent_idx in np.arange(env.N_AGENTS):
                # Append replay buffer
                uni_agent.store_memory(curr_grid_state, 
                                    curr_metadata_state, 
                                    actions[agent_idx], 
                                    probs[agent_idx],
                                    vals[agent_idx],
                                    rewards[agent_idx], 
                                    done
                                    )

            # Learning
            if step_count % learning_steps == 0 and step_count > uni_agent.batch_size:
                loss_t1 = uni_agent.learn().detach().numpy()
                loss_t2 = 0.0
            else:
                loss_t1 = 0.0
                loss_t2 = 0.0

            # Termination -> Append metrics
            if done or episode_step_count > max_steps:
                training_metrics['losses'].append((loss_t1, loss_t2))   
                training_metrics['team_1_captures'].append(env.metrics['team_points'][0])
                training_metrics['team_2_captures'].append(env.metrics['team_points'][1])
                training_metrics['team_1_tags'].append(env.metrics['tag_count'][0])
                training_metrics['team_2_tags'].append(env.metrics['tag_count'][1])
                for agent_idx in range(env.N_AGENTS):
                    training_metrics["agent_tag_count"][agent_idx].append(env.metrics['agent_tag_count'][agent_idx])
                    training_metrics["agent_flag_captures"][agent_idx].append(env.metrics['agent_flag_captures'][agent_idx])
                    training_metrics["agent_blocks_laid"][agent_idx].append(env.metrics['agent_blocks_laid'][agent_idx])
                    training_metrics["agent_blocks_mined"][agent_idx].append(env.metrics['agent_blocks_mined'][agent_idx])
                    training_metrics["agent_avg_distance_to_own_flag"][agent_idx].append(env.metrics['agent_total_distance_to_own_flag'][agent_idx]/env.env_step_count)
                    training_metrics["agent_avg_distance_to_opp_flag"][agent_idx].append(env.metrics['agent_total_distance_to_opp_flag'][agent_idx]/env.env_step_count)
                    training_metrics["agent_health_pickups"][agent_idx].append(env.metrics['agent_health_pickups'][agent_idx])

                done_count += 1 * done
                done = True
                
        training_metrics['score_history'].append(score)
        training_metrics['episode_step_counts'].append(episode_step_count)

        if i % 5 == 0:
            clear_output(wait=True)
            print(f"episode: {i+1} \
                \ntotal step count: {step_count} \
                \nepisode step count: {episode_step_count} \
                \nscore: {score} \
                \naverage score: {np.mean(training_metrics['score_history'][-100:])} \
                \ndone count: {done_count} \
                \nteam 1 captures: {sum(training_metrics['team_1_captures'])} \
                \nteam 2 captures: {sum(training_metrics['team_2_captures'])} \
                \nagent actions: {actions}" )

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

    return training_metrics
