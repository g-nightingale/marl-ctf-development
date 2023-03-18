import numpy as np
from mappo import PPOAgent
from gridworld_ctf_mvp import GridworldCtf
from collections import defaultdict
import torch as T
from IPython.display import clear_output
import utils as ut

def train_ppo(env,
            agent_t1,
            agent_t2,
            n_episodes,
            n_actors,
            learning_steps=20,
            max_steps=1000,
            use_ego_state=False,
            verbose_episodes=5,
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
        for _ in range(n_actors):
            env.reset()
            done = False
            episode_step_count = 0
            score = 0

            while not done: 
                step_count += 1
                episode_step_count += 1

                # Collect actions for each agent
                actions = []
                probs = []
                vals = []
                for agent_idx in np.arange(env.N_AGENTS):
                    # Get global and local states
                    metadata_state_local_ = env.get_env_metadata_local(agent_idx) 
                    metadata_state_local = T.from_numpy(metadata_state_local_).float().to(device)
                    
                    # Get global and local states
                    grid_state_local_ = env.standardise_state(agent_idx, use_ego_state=True)
                    grid_state_local = T.from_numpy(grid_state_local_).float().to(device)

                    #curr_grid_state = env.standardise_state(agent_idx, use_ego_state=use_ego_state, scale_tiles=scale_tiles).reshape(*env_dims) + ut.add_noise(env_dims)

                    if env.AGENT_TEAMS[agent_idx]==0:
                        action, prob = agent_t1.choose_action(grid_state_local, metadata_state_local)
                    else:
                        action, prob = agent_t2.choose_action(grid_state_local, metadata_state_local)

                    # Append actions and probs
                    actions.append(action)
                    probs.append(prob)

                # Step the environment
                _, rewards, done = env.step(actions)

                # Increment score
                score += sum(rewards)

                # Create the global metadata state: state + actions
                 # Get global metadata state
                metadata_state_global_ = env.get_env_metadata_global(actions)
                metadata_state_global = T.from_numpy(metadata_state_global_).float().to(device)

                for agent_idx in np.arange(env.N_AGENTS):
                    grid_state_global_ = env.standardise_state(agent_idx, use_ego_state=False)
                    grid_state_global = T.from_numpy(grid_state_global_).float().to(device)

                    if env.AGENT_TEAMS[agent_idx]==0:
                        val = agent_t1.get_state_value(grid_state_global, metadata_state_global)
                    else:
                        val = agent_t2.get_state_value(grid_state_global, metadata_state_global)
                    
                    vals.append(val)

                # Store each agent experiences
                for agent_idx in np.arange(env.N_AGENTS):
                    # Append replay buffer
                    if env.AGENT_TEAMS[agent_idx]==0:
                        agent_t1.store_memory(grid_state_local,
                                            grid_state_global, 
                                            metadata_state_local,
                                            metadata_state_global, 
                                            actions[agent_idx], 
                                            probs[agent_idx],
                                            vals[agent_idx],
                                            rewards[agent_idx], 
                                            done
                                            )
                    else:
                        agent_t2.store_memory(grid_state_local,
                                            grid_state_global, 
                                            metadata_state_local,
                                            metadata_state_global,
                                            actions[agent_idx], 
                                            probs[agent_idx],
                                            vals[agent_idx],
                                            rewards[agent_idx], 
                                            done
                                            )

                    if done or episode_step_count > max_steps:
                        done_count += 1 * done
                        done = True

        # Learning
        if step_count % learning_steps == 0 and step_count > agent_t1.batch_size:
            loss_t1 = agent_t1.learn().detach().numpy()
            loss_t2 = agent_t2.learn().detach().numpy()
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

        training_metrics['score_history'].append(score)
        training_metrics['episode_step_counts'].append(episode_step_count)

        if i % verbose_episodes == 0:
            clear_output(wait=True)
            print(f"episode: {i+1} \ttotal step count: {step_count} \tepisode step count: {episode_step_count} \tscore: {score} \taverage score: {np.mean(training_metrics['score_history'][-100:])} \tdone count: {done_count}")

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

    return training_metrics


if __name__ == '__main__':
    env = GridworldCtf(GAME_MODE='random',
                    GRID_SIZE=6,
                    AGENT_CONFIG = {
                    0: {'team':0, 'type':0},
                    1: {'team':1, 'type':0},
                    },
                    DROP_FLAG_WHEN_NO_HP=False,
                    GLOBAL_REWARDS=False)

    env.WINNING_POINTS=1

    n_actions = 8
    n_actors = 4
    n_episodes = 100
    learning_steps = 1
    max_steps = 256
    batch_size = 32
    n_epochs = 5
    alpha = 0.000003
    policy_clip = 0.1
    softmax_temp = 0.9
    use_ego_state = True

    verbose_episodes = 1
    local_grid_dims, global_grid_dims, local_metadata_dims, global_metadata_dims = env.get_env_dims()

    agent_t1 = PPOAgent(n_actions=n_actions, 
                        actor_grid_len=env.GRID_SIZE,
                        critic_grid_len=env.GRID_SIZE,
                        actor_metadata_len=local_metadata_dims[1], 
                        critic_metadata_len=global_metadata_dims[1],
                        batch_size=batch_size,
                        alpha=alpha, 
                        policy_clip=policy_clip,
                        n_epochs=n_epochs,
                        softmax_temp=softmax_temp)

    agent_t2 = PPOAgent(n_actions=n_actions, 
                        actor_grid_len=env.GRID_SIZE,
                        critic_grid_len=env.GRID_SIZE,
                        actor_metadata_len=local_metadata_dims[1], 
                        critic_metadata_len=global_metadata_dims[1],
                        batch_size=batch_size,
                        alpha=alpha, 
                        policy_clip=policy_clip,
                        n_epochs=n_epochs,
                        softmax_temp=softmax_temp)
    
    training_metrics = train_ppo(env,
                            agent_t1,
                            agent_t2,
                            n_episodes,
                            n_actors,
                            learning_steps=learning_steps,
                            max_steps=max_steps,
                            use_ego_state=use_ego_state,
                            verbose_episodes=verbose_episodes,
                            device='cpu')

    ut.plot_training_performance(training_metrics)
    
