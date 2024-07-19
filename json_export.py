from utils import get_agents, duel_json, TrainingConfig
from gridworld_ctf import GridworldCtf

def main():
    INFOLDER = 'runs/'
    OUTFOLDER = 'json/'
    FNAMES =[
        '0_the_split',
        '1_fence',
        '2_jailbreak',
        '3_one_way_out',
        '4_keyhole',
        '5_skittles',
        '6_the_wall',
        '7_gridlocked',
        '8_arena_iii'
    ]
    MAX_STEPS = 150
    TIMESTEPS = ['1', '20', '50']

    for fname in FNAMES:
        for t in TIMESTEPS:
            # Retrieve agents from time capsule
            env_config, agent1, agent2 = get_agents(INFOLDER + fname +'/',
                                                    f'time_capsule_0_{t}.bin',
                                                    0,
                                                    0,
                                                    0,
                                                    0)

            # Create environment with loaded config                         
            env = GridworldCtf(**env_config)

            # Pooduce JSON output
            _ = duel_json(env, 
                            agent1, 
                            agent2, 
                            max_steps=MAX_STEPS, 
                            render=False,
                            sleep_time=0.05,
                            fname=OUTFOLDER + fname + f'_{t}' + '.json')

if __name__ == '__main__':
    main()