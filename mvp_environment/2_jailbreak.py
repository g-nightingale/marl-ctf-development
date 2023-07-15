from league_training import LeagueTrainer
from scenarios import CtfScenarios as scn

class TrainingConfig():
    def __init__(self):
        #---------- Overall config
        self.wandb_project_name = "MARL-CTF-Test"
        self.exp_name = "2_jailbreak"
        self.use_wandb_selfplay = False
        self.use_wandb_ppo = False
        self.seed = 42
        self.checkpoint_frequency = 10
        self.number_of_metaruns = 10
        self.device = 'cpu'

        #---------- Self-play config
        self.number_of_iterations = 50
        self.number_of_duels = 30
        self.min_learning_rounds = 1
        self.n_main_agents = 1
        self.n_coaching_agents = 0
        self.n_league_agents = 0
        self.n_historical_agents = 0
        self.min_agent_winrate = 0.4
        self.min_historical_agent_winrate = 0.0
        self.min_agent_winrate_for_promotion = 0.6
        self.min_agent_iterations_for_replacement = 2
        self.inference_interval = 1
        self.historical_update_interval = 5

        #---------- Environment config
        self.env_config = {
            'GRID_SIZE':11,
            'AGENT_CONFIG':{
                0: {'team':0, 'type':0},
                1: {'team':1, 'type':0},
                2: {'team':0, 'type':3},
                3: {'team':1, 'type':0},
                # 4: {'team':0, 'type':2},
                # 5: {'team':1, 'type':2},
            },
            'SCENARIO': scn.jailbreak,
            'GAME_STEPS': 500,
            'USE_ADJUSTED_REWARDS': True,
            'MAP_SYMMETRY_CHECK': False
        }

        #---------- PPO Config
        self.n_actions = 9
        self.learning_rate = 0.0003
        self.total_timesteps = 12500
        self.torch_deterministic = True
        self.cuda = True
        self.wandb_entity = None
        self.parallel_rollouts = True
        self.num_envs = 8
        self.num_steps = 500
        self.anneal_lr = False
        self.gae = True
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.num_minibatches = 4
        self.update_epochs = 4
        self.norm_adv = False
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None


if __name__ == '__main__':
    args = TrainingConfig()
    league_trainer = LeagueTrainer(args)
    league_trainer.train_league()