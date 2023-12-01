from utils import get_agents, duel_json
from gridworld_ctf import GridworldCtf

def main():
    INFOLDER = 'runs/'
    OUTFOLDER = 'json/'
    FNAMES =[
        '0_the_split',
        '1_fence_ii',
        '2_jailbreak_ii',
        '3_one_way_out',
        '4_keyhole',
        '5_skittles',
        '6_the_wall_ii',
        '7_gridlocked',
        '8_arena_ii'
    ]

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
                            max_steps=500, 
                            render=False,
                            sleep_time=0.05,
                            fname=OUTFOLDER + fname + f'_{t}' + '.json')

if __name__ == '__main__':
    main()