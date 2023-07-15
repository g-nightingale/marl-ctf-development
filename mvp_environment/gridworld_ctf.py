import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import utils as ut
from collections import defaultdict
from functools import partial
import random
from PIL import Image
import os

class GridworldCtf:
    """
    A GridWorld capture the flag environment.
    
    """

    def __init__(self, 
                AGENT_CONFIG = {
                    0: {'team':0, 'type':0},
                    1: {'team':1, 'type':0},
                },
                SCENARIO=None,
                GAME_STEPS=256,
                GRID_SIZE=10,
                ENABLE_OBSTACLES=False,
                DROP_FLAG_WHEN_NO_HP=False,
                HOME_FLAG_CAPTURE=False,
                USE_EASY_CAPTURE=True,
                USE_ADJUSTED_REWARDS=False,
                MAX_BLOCK_TILE_PCT=0.2,
                LOG_METRICS=True,
                MAP_SYMMETRY_CHECK=True
                ):


        #----------------------------------------------------------------------------------
        # General Config
        #----------------------------------------------------------------------------------
        self.AGENT_CONFIG = AGENT_CONFIG
        self.SCENARIO = SCENARIO
        self.GAME_STEPS = GAME_STEPS
        self.GRID_SIZE = GRID_SIZE
        self.ENV_DIMS = (1, 1, GRID_SIZE, GRID_SIZE)
        self.ENABLE_OBSTACLES = ENABLE_OBSTACLES
        self.DROP_FLAG_WHEN_NO_HP = DROP_FLAG_WHEN_NO_HP
        self.HOME_FLAG_CAPTURE = HOME_FLAG_CAPTURE
        self.USE_EASY_CAPTURE = USE_EASY_CAPTURE
        self.MAX_BLOCK_TILE_PCT = MAX_BLOCK_TILE_PCT
        self.N_AGENTS = len(self.AGENT_CONFIG)
        self.LOG_METRICS = LOG_METRICS
        self.MAP_SYMMETRY_CHECK = MAP_SYMMETRY_CHECK
        self.ACTION_SPACE = 8
        self.FLIP_AXIS = None

        # Rewards
        self.WIN_MARGIN_SCALAR = 0.1
        self.LOSS_MARGIN_SCALAR = 0.00
        self.REWARD_CAPTURE = 1
        self.REWARD_STEP = 0
        self.REWARD_TAG = 0.0
        self.OPP_FLAG_CAPTURE_PUNISHMENT_SCALAR = 0.5
        self.WINNING_POINTS = np.inf
        self.USE_ADJUSTED_REWARDS = USE_ADJUSTED_REWARDS

        # Block counts for constructor agents
        self.BLOCK_PICKUP_VALUE = 1
        self.HEALTH_PICKUP_VALUE = 1
        self.DEFENSIVE_ZONE_DISTANCE = 3

        # Action mapping
        # U1 -> 0 
        # D1 -> 1 
        # R1 -> 2 
        # L1 -> 3 
        # NM -> 4 
        # U2 -> 5 
        # D2 -> 6 
        # R2 -> 7 
        # L2 -> 8
        #  
        self.ACTION_DELTAS = {
                0: {
                    0: (-1, 0),
                    1: (1, 0),
                    2: (0, 1),
                    3: (0, -1),
                    4: (0, 0),
                    5: (0, 0),
                    6: (0, 0),
                    7: (0, 0),
                    8: (0, 0)
                },
                1: {
                    0: (-1, 0),
                    1: (1, 0),
                    2: (0, 1),
                    3: (0, -1),
                    4: (0, 0),
                    5: (0, 0),
                    6: (0, 0),
                    7: (0, 0),
                    8: (0, 0)
                },
                2: {
                    0: (-1, 0),
                    1: (1, 0),
                    2: (0, 1),
                    3: (0, -1),
                    4: (0, 0),
                    5: (-2, 0),
                    6: (2, 0),
                    7: (0, 2),
                    8: (0, -2)
                },
                3: {
                    0: (-1, 0),
                    1: (1, 0),
                    2: (0, 1),
                    3: (0, -1),
                    4: (0, 0),
                    5: (-1, 0),
                    6: (1, 0),
                    7: (0, 1),
                    8: (0, -1)
                },
        }

        self.REVERSED_ACTION_MAP = {
            # Both axis flipped
            None: {
                    0: 1,
                    1: 0,
                    2: 3,
                    3: 2,
                    4: 4,
                    5: 6,
                    6: 5,
                    7: 8,
                    8: 7
            },
            # Vertical axis flipped
            0: {
                    0: 1,
                    1: 0,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 6,
                    6: 5,
                    7: 7,
                    8: 8
            },
            # Horizontal axis flipped
            1: {
                    0: 0,
                    1: 1,
                    2: 3,
                    3: 2,
                    4: 4,
                    5: 5,
                    6: 6,
                    7: 8,
                    8: 7
            },
            # Diagonal flipped
            2: {
                    0: 2,
                    1: 3,
                    2: 0,
                    3: 1,
                    4: 4,
                    5: 7,
                    6: 8,
                    7: 5,
                    8: 6
            }
        }

        #----------------------------------------------------------------------------------
        # Agent Config
        #----------------------------------------------------------------------------------

        # Teams that agent's belong to: 0 = team 1, 1 = team 2
        self.AGENT_TEAMS = {k:self.AGENT_CONFIG[k]['team'] for k in self.AGENT_CONFIG.keys()}

        # Agent types: 0 = scout, 1 = vaulter, 2 = guardian, 3 = miner
        self.AGENT_TYPES = {k:self.AGENT_CONFIG[k]['type'] for k in self.AGENT_CONFIG.keys()}

        # Agent hitpoints
        self.AGENT_TYPE_HP = {
            0: 8,
            1: 6,
            2: 4,
            3: 4
        }

        # HP healing per step
        self.AGENT_HP_HEALING_PER_STEP = 0.25

        # Damage dealt by each agent type
        self.AGENT_TYPE_DAMAGE = {
            0: 1,
            1: 0.5,
            2: 1,
            3: 1
        }

        # Action masks
        self.AGENT_TYPE_ACTION_MASK = {
            0: 1,
            1: 1,
            2: 0,
            3: 0
        }

        # Distance at which guardian has defense capability
        self.GUARDIAN_DEFENSE_DISTANCE = 3
        self.GUARDIAN_TAGGING_RANGE = 1
        self.GUARDIAN_DAMAGE_MULTIPLIER = 5.0

        # Probability of a successful tag for guardian agent
        self.TAG_PROBABILITY = 0.75

        # Vaulter hitpoint cost to make a jump
        self.VAULT_HP_COST = 0.5
        self.VAULT_MIN_HP = 2.5

        # Agent flag capture types - types that can capture the flag
        self.AGENT_FLAG_CAPTURE_TYPES = [0, 1, 2, 3]

        # Maximum number of blocks the miner agent can carry
        self.MAX_AGENT_BLOCKS = 1000

        # self._arr = np.arange(self.N_AGENTS, dtype=np.uint8)
        self._arr = [x for x in range(self.N_AGENTS)]

        #----------------------------------------------------------------------------------
        # Tile definitions
        #----------------------------------------------------------------------------------
        # Open and block tiles
        self.OPEN_TILE = 0
        self.BLOCK_TILE = 1
        self.DESTRUCTIBLE_TILE1 = 2
        self.DESTRUCTIBLE_TILE2 = 3

        # Tile values for each agent type by team
        self.AGENT_TYPE_TILE_MAP = {
            0: {0:4, 1:8},
            1: {0:5, 1:9},
            2: {0:6, 1:10},
            3: {0:7, 1:11},
        }

        # Derive AGENT_TILE_MAP from AGENT_TYPE_TILE_MAP and AGENT_CONFIG
        self.AGENT_TILE_MAP = {k:self.AGENT_TYPE_TILE_MAP[AGENT_CONFIG[k]['type']][AGENT_CONFIG[k]['team']] for k in AGENT_CONFIG.keys()}

        # Tile values for flags
        self.FLAG_TILE_MAP = {
            0: 12,
            1: 13
        }

        # Standardised tiles
        self.STD_BLOCK_TILE = 1
        self.STD_DESTRUCTIBLE_TILE1 = 2
        self.STD_DESTRUCTIBLE_TILE2 = 3

        self.STD_AGENT_TYPE_TILES_OWN = {
            0: 4,
            1: 5,
            2: 6,
            3: 7,
        }
        self.STD_AGENT_TYPE_TILES_OPP = {
            0: 8,
            1: 9,
            2: 10,
            3: 11,
        }

        self.STD_OWN_FLAG_TILE = 12
        self.STD_OPP_FLAG_TILE = 13

        #----------------------------------------------------------------------------------
        # Colour map for rendering environment
        #----------------------------------------------------------------------------------
        # See: https://www.rapidtables.com/web/color/RGB_Color.html
        
        self.COLOUR_MAP = {
                    0: np.array([224, 224, 224]), # Open tile
                    1: np.array([0, 0, 0]), # Block tile
                    2: np.array([32, 32, 32]), # Destructible tile 1
                    3: np.array([64, 64, 64]), # Destructible tile 2
                    4: np.array([102, 178, 255]), # Blue team agent type 1
                    5: np.array([0, 128, 255]), # Blue team agent type 2
                    6: np.array([153, 204, 255]), # Blue team agent type 3
                    7: np.array([153, 210, 255]), # Blue team agent type 4
                    8: np.array([255, 102, 102]), # Red team agent type 1
                    9: np.array([255, 51, 51]), # Red team agent type 2
                    10: np.array([255, 153, 153]), # Red team agent type 3
                    11: np.array([255, 163, 153]), # Red team agent type 4
                    12: np.array([0, 0, 153]), # Blue team flag
                    13: np.array([153, 0, 0]), # Red team flag
        }  


        #----------------------------------------------------------------------------------
        # Images rendering environment
        #----------------------------------------------------------------------------------
        IMAGE_FOLDER_PATH = os.getcwd() + '/img/'

        # Load images
        self.IMAGE_MAP = {
            0: Image.open(IMAGE_FOLDER_PATH + 'tile0.png'),
            1: Image.open(IMAGE_FOLDER_PATH + 'tile1.png'),
            2: Image.open(IMAGE_FOLDER_PATH + 'tile2.png'),
            3: Image.open(IMAGE_FOLDER_PATH + 'tile3.png'),
            4: Image.open(IMAGE_FOLDER_PATH + 'scout_blue.png'),
            5: Image.open(IMAGE_FOLDER_PATH + 'guardian_blue.png'),
            6: Image.open(IMAGE_FOLDER_PATH + 'vaulter_blue.png'),
            7: Image.open(IMAGE_FOLDER_PATH + 'miner_blue.png'),
            8: Image.open(IMAGE_FOLDER_PATH + 'scout_red.png'),
            9: Image.open(IMAGE_FOLDER_PATH + 'guardian_red.png'),
            10: Image.open(IMAGE_FOLDER_PATH + 'vaulter_red.png'),
            11: Image.open(IMAGE_FOLDER_PATH + 'miner_red.png'),
            12: Image.open(IMAGE_FOLDER_PATH + 'flag_blue.png'),
            13: Image.open(IMAGE_FOLDER_PATH + 'flag_red.png'),
            104: Image.open(IMAGE_FOLDER_PATH + 'scout_blue_flag.png'),
            105: Image.open(IMAGE_FOLDER_PATH + 'guardian_blue_flag.png'),
            106: Image.open(IMAGE_FOLDER_PATH + 'vaulter_blue_flag.png'),
            107: Image.open(IMAGE_FOLDER_PATH + 'miner_blue_flag.png'),
            108: Image.open(IMAGE_FOLDER_PATH + 'scout_red_flag.png'),
            109: Image.open(IMAGE_FOLDER_PATH + 'guardian_red_flag.png'),
            110: Image.open(IMAGE_FOLDER_PATH + 'vaulter_red_flag.png'),
            111: Image.open(IMAGE_FOLDER_PATH + 'miner_red_flag.png'),
            112: Image.open(IMAGE_FOLDER_PATH + 'flag_blue_empty.png'),
            113: Image.open(IMAGE_FOLDER_PATH + 'flag_red_empty.png'),
        }

        # Reset the environment
        self.reset()

    def load_scenario(self, scenario):
        """
        Load a fixed gameplay scenario.
        """
        # Load attributes
        self.SCENARIO_NAME = scenario['SCENARIO_NAME']
        self.FLIP_AXIS = scenario['FLIP_AXIS']
        self.GRID_SIZE = scenario['GRID_SIZE']
        self.FLAG_POSITIONS = scenario['FLAG_POSITIONS']
        self.CAPTURE_POSITIONS = scenario['CAPTURE_POSITIONS']
        self.SPAWN_POSITIONS = scenario['SPAWN_POSITIONS']
        self.AGENT_STARTING_POSITIONS = scenario['AGENT_STARTING_POSITIONS']

        # Initialise grid
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        for slc in scenario['BLOCK_TILE_SLICES']:
            grid[slc] = self.BLOCK_TILE

        for slc in scenario['DESTRUCTIBLE_TILE_SLICES']:
            grid[slc] = self.DESTRUCTIBLE_TILE1

        # Add flags
        grid[self.FLAG_POSITIONS[0]] = self.FLAG_TILE_MAP[0]
        grid[self.FLAG_POSITIONS[1]] = self.FLAG_TILE_MAP[1]

        # Add agents
        for agent_idx in range(self.N_AGENTS):
            grid[self.AGENT_STARTING_POSITIONS[agent_idx]] = self.AGENT_TILE_MAP[agent_idx]

        self.grid = grid

    def reset(self):
        """ 
        Reset the environment. 
        """

        self.env_step_count = 0
        self.done = False

        agents_per_team = self.N_AGENTS // 2
        self.OPPONENTS = {
            0: [k for k, v in self.AGENT_TEAMS.items() if v==1][:agents_per_team],
            1: [k for k, v in self.AGENT_TEAMS.items() if v==0][:agents_per_team],
        }
        
        # Generate the map
        if self.SCENARIO is None:
            self.generate_map()
        else:
            self.load_scenario(self.SCENARIO)

        # Tiles used in environment
        self.TILES_USED = self.get_tiles_used()

        # Get agent positions
        self.agent_positions = {k:v for k, v in self.AGENT_STARTING_POSITIONS.items() if k < self.N_AGENTS}

        # Game metadata
        self.has_flag = np.zeros(self.N_AGENTS, dtype=np.uint8)
        self.steps_with_flag = np.zeros(self.N_AGENTS, dtype=np.uint8)
        self.agent_teams_np = np.array([v for v in self.AGENT_TEAMS.values()], dtype=np.uint8)
        
        # Agent hitpoints
        self.agent_hp = {kv[0]:self.AGENT_TYPE_HP[kv[1]] for kv in self.AGENT_TYPES.items()}

        # Agent vault power
        self.block_inventory = {kv[0]:0 if kv[1]==2 else 0 for kv in self.AGENT_TYPES.items()}

        # Agent block inventory
        self.block_inventory = {kv[0]:0 if kv[1]==3 else 0 for kv in self.AGENT_TYPES.items()}

        # Internal var to track flag capture in current move
        self._flag_capture_current_move = False
        self._flag_capture_team_current_move = {0:0, 1:0}

        # Metrics dictionary
        self.metrics = {
            # Team level metrics
            "team_wins": {0:0, 1:0},
            "team_tag_count": {0:0, 1:0},
            "team_respawn_tag_count": {0:0, 1:0},
            "team_flag_pickups": {0:0, 1:0},
            "team_flag_captures": {0:0, 1:0},
            "team_flag_dispossessions": {0:0, 1:0},
            "team_blocks_laid": {0:0, 1:0},
            "team_blocks_mined": {0:0, 1:0},
            "team_blocks_laid_distance_from_own_flag": {0:0, 1:0},
            "team_blocks_laid_distance_from_opp_flag": {0:0, 1:0},
            "team_steps_defending_zone": {0:0, 1:0},
            "team_steps_attacking_zone": {0:0, 1:0},
            "team_steps_adj_teammate": {0:0, 1:0},
            "team_steps_adj_opponent": {0:0, 1:0},
            # Agent type metrics
            "agent_type_tag_count": defaultdict(lambda: defaultdict(int)),
            "agent_type_respawn_tag_count": defaultdict(lambda: defaultdict(int)),
            "agent_type_flag_pickups": defaultdict(lambda: defaultdict(int)),
            "agent_type_flag_captures": defaultdict(lambda: defaultdict(int)),
            "agent_type_flag_dispossessions": defaultdict(lambda: defaultdict(int)),
            "agent_type_blocks_laid": defaultdict(lambda: defaultdict(int)),
            "agent_type_blocks_mined": defaultdict(lambda: defaultdict(int)),
            "agent_type_blocks_laid_distance_from_own_flag": defaultdict(lambda: defaultdict(int)),
            "agent_type_blocks_laid_distance_from_opp_flag": defaultdict(lambda: defaultdict(int)),
            "agent_type_steps_defending_zone": defaultdict(lambda: defaultdict(int)),
            "agent_type_steps_attacking_zone": defaultdict(lambda: defaultdict(int)),
            "agent_type_steps_adj_teammate": defaultdict(lambda: defaultdict(int)),
            "agent_type_steps_adj_opponent": defaultdict(lambda: defaultdict(int)),
            # Agent level metrics
            "agent_tag_count": defaultdict(int),
            "agent_respawn_tag_count": defaultdict(int),
            "agent_flag_pickups": defaultdict(int),
            "agent_flag_captures": defaultdict(int),
            "agent_flag_dispossessions": defaultdict(int),
            "agent_blocks_laid": defaultdict(int),
            "agent_blocks_mined": defaultdict(int),
            "agent_blocks_laid_distance_from_own_flag": defaultdict(int),
            "agent_blocks_laid_distance_from_opp_flag": defaultdict(int),
            "agent_steps_defending_zone": defaultdict(int),
            "agent_steps_attacking_zone": defaultdict(int),
            "agent_steps_adj_teammate": defaultdict(int),
            "agent_steps_adj_opponent": defaultdict(int),
            "agent_visitation_maps": defaultdict(lambda: np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)),
        }

        # Update the visitation maps with starting positions
        self.update_visitation_map()

        # Check symmetry of map
        if self.MAP_SYMMETRY_CHECK:
            assert(np.all(self.standardise_state(0) == self.standardise_state(1, reverse_grid=True)))

    def update_visitation_map(self):
        """
        Update the agent visitation maps.
        """
        # Init the visitation maps
        for agent_idx in range(self.N_AGENTS):
            agent_pos = self.agent_positions[agent_idx]
            self.metrics["agent_visitation_maps"][agent_idx][agent_pos[0], agent_pos[1]] += 1

    def get_tiles_used(self):
        """
        Get the tiles used by the current env configuration.
        Used to standardise the state across multiple channels.
        """
        
        tiles_list = [x for x in np.unique(self.grid) if x!=0]
        tiles_list.extend([self.DESTRUCTIBLE_TILE2]) if self.DESTRUCTIBLE_TILE1 in tiles_list and 3 in self.AGENT_TYPES.values() else None
        tiles_list += [self.STD_AGENT_TYPE_TILES_OPP[agent_type] for agent_type in self.AGENT_TYPES.values()]
        tiles_list = list(set(tiles_list))

        return tiles_list

    def generate_map(self):
        """
        Randomly generate a map.
        """

        # Initialise grid
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        # Decide flag positions zones
        t1_y = np.random.choice(np.arange(1, self.GRID_SIZE-1))
        t2_y = np.random.choice(np.arange(1, self.GRID_SIZE-1))

        self.FLAG_POSITIONS[0] = (t1_y, 1)
        self.FLAG_POSITIONS[1] = (t2_y, self.GRID_SIZE-2)
        
        t1_flag_pos = self.FLAG_POSITIONS[0]
        t2_flag_pos = self.FLAG_POSITIONS[1]

        # Define spawn positions
        self.SPAWN_POSITIONS = self.FLAG_POSITIONS.copy()

        # Place obstacles
        if self.ENABLE_OBSTACLES:
            # Set all tiles to destructible tile 1
            grid[:, :] = self.DESTRUCTIBLE_TILE1

            # Clear flag + spawning zones
            grid[t1_flag_pos[0]-1:t1_flag_pos[0]+2, t1_flag_pos[1]-1:t1_flag_pos[1]+2] = self.OPEN_TILE
            grid[t2_flag_pos[0]-1:t2_flag_pos[0]+2, t2_flag_pos[1]-1:t2_flag_pos[1]+2] = self.OPEN_TILE

            tile_removal_counter = 0
            # Keep clearing blocks one by one
            while (grid==self.OPEN_TILE).sum() < self.GRID_SIZE**2 * (1 - self.MAX_BLOCK_TILE_PCT):
                tile_removal_counter += 1
                x = np.random.randint(0, self.GRID_SIZE)
                y = np.random.randint(0, self.GRID_SIZE)
                grid[x, y] = self.OPEN_TILE

                if tile_removal_counter > 1000:
                    break

        # Add flags
        grid[self.FLAG_POSITIONS[0]] = self.FLAG_TILE_MAP[0]
        grid[self.FLAG_POSITIONS[1]] = self.FLAG_TILE_MAP[1]

        # Add agents
        for agent_idx in range(self.N_AGENTS):
            # Get agent team flag position - we will respawn around there
            if self.AGENT_TEAMS[agent_idx]==0:
                x, y = t1_flag_pos
            else:
                x, y = t2_flag_pos

            possible_respawn_offset = np.where(grid[max(x-1, 0):x+2, max(y-1, 0):y+2]==self.OPEN_TILE)

            # Choose a free respawn location at random
            rnd = np.random.randint(possible_respawn_offset[0].shape[0])
            
            # Get respawn position: WARNING - if the flag is at (0, 0) this will break 
            new_pos = (x + possible_respawn_offset[0][rnd] - 1, y + possible_respawn_offset[1][rnd] - 1)

            # Update tiles
            grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]

            self.AGENT_STARTING_POSITIONS[agent_idx] = new_pos
        
        self.grid = grid

    def movement_handler(self, agent_idx, curr_pos, new_pos):
        """
        Handles agent movements and block tile damage.
        """

        #----------------------------------------------------------------------------------
        # Movement into a free tile
        #----------------------------------------------------------------------------------
        if self.grid[new_pos] == self.OPEN_TILE:
            self.grid[curr_pos] = 0
            self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
            curr_pos = new_pos

            # Flag pickup
            if self.max_dim_distance_to_xy(new_pos, self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]) <= 1 \
              and self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]]==self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]] \
              and self.AGENT_TYPES[agent_idx] in self.AGENT_FLAG_CAPTURE_TYPES:
                self.has_flag[agent_idx] = 1
                self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.BLOCK_TILE
                # Update metrics
                self.metrics['team_flag_pickups'][self.AGENT_TEAMS[agent_idx]] += 1
                self.metrics['agent_type_flag_pickups'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_flag_pickups'][agent_idx] += 1

            # Flag capture
            if self.max_dim_distance_to_xy(new_pos, self.FLAG_POSITIONS[self.AGENT_TEAMS[agent_idx]]) <= 1 \
              and self.has_flag[agent_idx] == 1:
                # Check if own flag needs to be 'at home' to capture
                if (self.HOME_FLAG_CAPTURE and self.grid[self.FLAG_POSITIONS[self.AGENT_TEAMS[agent_idx]]]==self.FLAG_TILE_MAP[self.AGENT_TEAMS[agent_idx]]) \
                    or not self.HOME_FLAG_CAPTURE:
                    self.has_flag[agent_idx] = 0
                    self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]
                    # Update metrics
                    self.metrics['team_flag_captures'][self.AGENT_TEAMS[agent_idx]] += 1
                    self.metrics['agent_type_flag_captures'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                    self.metrics['agent_flag_captures'][agent_idx] += 1
                    # Check for win
                    if self.metrics['team_wins'][self.AGENT_TEAMS[agent_idx]] == self.WINNING_POINTS:
                        self.done = True
                    # Update indicator
                    self._flag_capture_current_move = True
                    self._flag_capture_team_current_move[self.AGENT_TEAMS[agent_idx]] = 1.0

        return curr_pos

    def add_block(self, pos, agent_idx):
        """
        Add a block to the environment for the miner agent class.
        """
        self.grid[pos] = self.DESTRUCTIBLE_TILE1
        self.block_inventory[agent_idx] -= 1

        # Update metrics
        distance_to_own_flag = self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[self.AGENT_TEAMS[agent_idx]])
        distance_to_opp_flag = self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[1 - self.AGENT_TEAMS[agent_idx]])

        self.metrics['team_blocks_laid'][self.AGENT_TEAMS[agent_idx]] += 1
        self.metrics['agent_type_blocks_laid'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
        self.metrics['agent_blocks_laid'][agent_idx] += 1
        
        self.metrics['team_blocks_laid_distance_from_own_flag'][self.AGENT_TEAMS[agent_idx]] += distance_to_own_flag
        self.metrics['agent_type_blocks_laid_distance_from_own_flag'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += distance_to_own_flag
        self.metrics['agent_blocks_laid_distance_from_own_flag'][agent_idx] += distance_to_own_flag
        self.metrics['team_blocks_laid_distance_from_opp_flag'][self.AGENT_TEAMS[agent_idx]] += distance_to_opp_flag
        self.metrics['agent_type_blocks_laid_distance_from_opp_flag'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += distance_to_opp_flag
        self.metrics['agent_blocks_laid_distance_from_opp_flag'][agent_idx] += distance_to_opp_flag

    def is_valid_move(self, new_pos):
        """
        Check the move is valid.
        """
        x, y = new_pos
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE
    
    def move_to_open_tile(self, agent_idx, action, new_pos):
        """
        Check if move is to an open tile.
        """
        return self.grid[new_pos] == self.OPEN_TILE \
                and ((action <= 3) 
                     or (action >=5 and self.AGENT_TYPES[agent_idx] == 2 
                         and (self.agent_hp[agent_idx] - self.VAULT_HP_COST) > self.VAULT_MIN_HP))
    
    def update_vaulter_hp(self, agent_idx, action):
        """
        Reduce vaulter HP by vault HP cost.
        """
        if action >=5 and self.AGENT_TYPES[agent_idx] == 2:
            self.agent_hp[agent_idx] -= self.VAULT_HP_COST

    def can_add_blocks(self, agent_idx, action, new_pos):
        """
        Check if conditions are met to add a block.
        """
        return action >= 5 and self.AGENT_TYPES[agent_idx] == 3 \
                and self.block_inventory[agent_idx] > 0 \
                and self.grid[new_pos] == self.OPEN_TILE
    
    def can_mine_blocks(self, agent_idx, new_pos):
        """
        Check conditions for mining blocks.
        """
        return self.AGENT_TYPES[agent_idx] == 3 \
                and (self.grid[new_pos] == self.DESTRUCTIBLE_TILE1 \
                or self.grid[new_pos] == self.DESTRUCTIBLE_TILE2)
    
    def mine_block(self, agent_idx, new_pos):
        """
        Mine block.
        """
        if self.grid[new_pos] == self.DESTRUCTIBLE_TILE1:
            self.grid[new_pos] = self.DESTRUCTIBLE_TILE2
        elif self.grid[new_pos] == self.DESTRUCTIBLE_TILE2:
            self.grid[new_pos] = self.OPEN_TILE
            if self.block_inventory[agent_idx] < self.MAX_AGENT_BLOCKS:
                self.block_inventory[agent_idx] += self.BLOCK_PICKUP_VALUE
            # Update metrics
            self.metrics['team_blocks_mined'][self.AGENT_TEAMS[agent_idx]] += 1
            self.metrics['agent_type_blocks_mined'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
            self.metrics['agent_blocks_mined'][agent_idx] += 1

    def reset_flag(self, agent_idx):
        """
        Reset flag if agent has been holding it for too long.
        """
        self.has_flag[agent_idx] = 0
        self.steps_with_flag[agent_idx] = 0
        self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]

    def act(self, agent_idx, action) -> None:
        """
        Take agent actions and update grid.
        """

        # Init reward
        reward = 0

        # Get the current and new position of the agent
        curr_pos = self.agent_positions[agent_idx]
        action_delta = self.ACTION_DELTAS[self.AGENT_TYPES[agent_idx]][action]
        new_pos = (curr_pos[0] + action_delta[0], curr_pos[1] + action_delta[1])

        # Check if move into new cell is valid
        if self.is_valid_move(new_pos):
            if self.move_to_open_tile(agent_idx, action, new_pos):
                curr_pos = self.movement_handler(agent_idx, curr_pos, new_pos)
                self.update_vaulter_hp(agent_idx, action)
            elif self.can_add_blocks(agent_idx, action, new_pos):
                self.add_block(new_pos, agent_idx)
            elif self.can_mine_blocks(agent_idx, new_pos):
                self.mine_block(agent_idx, new_pos)

        # Update agent position
        self.agent_positions[agent_idx] = curr_pos

        # Step reward is applied in all situations
        reward += self.REWARD_STEP
        if self._flag_capture_current_move:
            reward += self.REWARD_CAPTURE
            self._flag_capture_current_move = False

        return reward
     
    def dice_roll(self) -> np.array:
        """
        Determines order that agents make moves.
        """

        # np.random.shuffle(self._arr)
        random.shuffle(self._arr)

        return self._arr

    def agent_distance_to_xy(self, agent_idx, object_xy) -> bool:
        """
        Returns agent distance to a given x, y on the grid.
        """
        
        x, y = self.agent_positions[agent_idx]
        distances = np.abs(np.array([x, y]) - np.array([object_xy[0], object_xy[1]]))
        return max(distances)

    def max_dim_distance_to_xy(self, xy, target_xy):
        """
        Returns maximum dimension distance between two points on a grid.
        """
        
        distances = np.abs(np.array([xy[0], xy[1]]) - np.array([target_xy[0], target_xy[1]]))
        return max(distances)
    
    def respawn(self, agent_idx):
        """
        Respawn agent near flag if tagged.
        """

        # Get agent team flag position - we will respawn around there
        x, y = self.SPAWN_POSITIONS[self.AGENT_TEAMS[agent_idx]]
        possible_respawn_offset = np.where(self.grid[max(x-1, 0):x+2, max(y-1, 0):y+2]==self.OPEN_TILE)

        # Choose a free respawn location at random
        rnd = np.random.randint(possible_respawn_offset[0].shape[0])
        
        # Get respawn position: WARNING - if the flag is at (0, 0) this will break 
        old_pos = self.agent_positions[agent_idx]
        new_pos = (x + possible_respawn_offset[0][rnd] - 1, y + possible_respawn_offset[1][rnd] - 1)

        # Update tiles
        self.grid[old_pos] = self.OPEN_TILE
        self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
        
        # Update agent position
        self.agent_positions[agent_idx] = new_pos

        # Reset agent hp
        self.agent_hp[agent_idx] = self.AGENT_TYPE_HP[self.AGENT_TYPES[agent_idx]]

        # Reset flag if agent has it
        if self.has_flag[agent_idx]==1:
            self.has_flag[agent_idx] = 0
            self.steps_with_flag[agent_idx] = 0
            if self.DROP_FLAG_WHEN_NO_HP:
                self.grid[old_pos] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]
            else:
                self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]

    def tagging_logic(self, agent_idx):
        """
        Apply tagging logic.
        """

        tagging_reward = 0

        # Check that agent has guardian ability and is within defense zone
        if self.AGENT_TYPE_DAMAGE[self.AGENT_TYPES[agent_idx]] > 0:
            
            # Calculate damage multiplier
            DAMAGE_MULTIPLIER = 1.0
            if self.agent_distance_to_xy(agent_idx, self.FLAG_POSITIONS[self.AGENT_TEAMS[agent_idx]]) <= self.GUARDIAN_DEFENSE_DISTANCE \
              and self.AGENT_TYPES[agent_idx] == 1:
                DAMAGE_MULTIPLIER = self.GUARDIAN_DAMAGE_MULTIPLIER

            # Loop through opponents
            for opp_agent_idx in self.OPPONENTS[self.AGENT_TEAMS[agent_idx]]:
                # Check opponent is within tagging range
                if np.random.rand() < self.TAG_PROBABILITY \
                    and self.agent_distance_to_xy(agent_idx, self.agent_positions[opp_agent_idx]) <= self.GUARDIAN_TAGGING_RANGE:
                    # Reduce opponent hit points by the amount of damage incurred by the agent type
                    self.agent_hp[opp_agent_idx] -= self.AGENT_TYPE_DAMAGE[self.AGENT_TYPES[agent_idx]] * DAMAGE_MULTIPLIER
                    # If hitpoints are zero, return flag and respawn opponent agent
                    # Update metrics
                    self.metrics['team_tag_count'][self.AGENT_TEAMS[agent_idx]]  += 1
                    self.metrics['agent_type_tag_count'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                    self.metrics['agent_tag_count'][agent_idx]  += 1
                    if self.agent_hp[opp_agent_idx] <= 0:
                        if self.has_flag[opp_agent_idx]==1:
                            self.metrics['team_flag_dispossessions'][self.AGENT_TEAMS[agent_idx]]  += 1
                            self.metrics['agent_type_flag_dispossessions'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                            self.metrics['agent_flag_dispossessions'][agent_idx]  += 1

                        # Respawn opponent agent (also handles flag return)
                        self.respawn(opp_agent_idx)
                        tagging_reward = self.REWARD_TAG
                        self.metrics['team_respawn_tag_count'][self.AGENT_TEAMS[agent_idx]] += 1
                        self.metrics['agent_type_respawn_tag_count'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                        self.metrics['agent_respawn_tag_count'][agent_idx]  += 1
                        
        return tagging_reward
    
    def heal_agents(self):
        """
        Heal agents each timestep.
        """

        for agent_idx in self.dice_roll():
            if self.agent_hp[agent_idx] < self.AGENT_TYPE_HP[self.AGENT_TYPES[agent_idx]]:
                self.agent_hp[agent_idx] += self.AGENT_HP_HEALING_PER_STEP
                self.agent_hp[agent_idx] = min(self.agent_hp[agent_idx], self.AGENT_TYPE_HP[self.AGENT_TYPES[agent_idx]])

    def step(self, actions):
        """
        Take a step in the environment.

        Takes in a vector or actions, where each position in the vector represents an agent,
        and the value represents the action to take.
        """

        self.env_step_count += 1
        self._flag_capture_team_current_move = {0:0, 1:0}

        rewards = [0] * self.N_AGENTS
        for agent_idx in self.dice_roll():

            # Get the agent team
            agent_team = self.AGENT_TEAMS[agent_idx]

            # Get the agent action
            action = actions[agent_idx]

            # Move the agent and get reward
            rewards[agent_idx] = self.act(agent_idx, action)

            # Check for tags
            rewards[agent_idx] += self.tagging_logic(agent_idx)

            #----------------------------------------------------------------------------------
            # Logging extra metrics
            #----------------------------------------------------------------------------------
            # Zonal metrics
            distance_to_own_flag = self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[agent_team])
            distance_to_opp_flag = self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[1 - agent_team])

            if distance_to_own_flag <= self.DEFENSIVE_ZONE_DISTANCE:
                self.metrics['team_steps_defending_zone'][agent_team] += 1
                self.metrics['agent_type_steps_defending_zone'][agent_team][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_steps_defending_zone'][agent_idx] += 1
            if distance_to_opp_flag <= self.DEFENSIVE_ZONE_DISTANCE:
                self.metrics['team_steps_attacking_zone'][agent_team] += 1
                self.metrics['agent_type_steps_attacking_zone'][agent_team][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_steps_attacking_zone'][agent_idx] += 1

            # Proximity metrics
            for teammate_idx in self.OPPONENTS[1-agent_team]:
                if self.agent_distance_to_xy(agent_idx, self.agent_positions[teammate_idx]) <= 1:
                    self.metrics['team_steps_adj_teammate'][agent_team] += 1
                    self.metrics['agent_type_steps_adj_teammate'][agent_team][self.AGENT_TYPES[agent_idx]] += 1
                    self.metrics['agent_steps_adj_teammate'][agent_idx] += 1

            for opp_agent_idx in self.OPPONENTS[agent_team]:
                if self.agent_distance_to_xy(agent_idx, self.agent_positions[opp_agent_idx]) <= 1:
                    self.metrics['team_steps_adj_opponent'][agent_team] += 1
                    self.metrics['agent_type_steps_adj_opponent'][agent_team][self.AGENT_TYPES[agent_idx]] += 1
                    self.metrics['agent_steps_adj_opponent'][agent_idx] += 1

        # Heal agents 
        self.heal_agents()

        if self.USE_ADJUSTED_REWARDS:
            rewards = self.get_adjusted_rewards(rewards)

        # Update the visitation maps with new positions
        self.update_visitation_map()

        # Check for end game state
        if self.env_step_count == self.GAME_STEPS:
            self.done = True
            rewards = self.get_terminal_rewards(rewards)

        return self.grid, rewards, self.done

    def get_terminal_rewards(self, rewards, penalise_losing_team=True):
        """
        Get terminal rewards for each team.
        """
        # Find the winning team
        winning_team = None
        score_margin = np.abs(self.metrics['team_flag_captures'][0] - self.metrics['team_flag_captures'][1])
        if self.metrics['team_flag_captures'][0] > self.metrics['team_flag_captures'][1]:
            winning_team = 0
        elif self.metrics['team_flag_captures'][0] < self.metrics['team_flag_captures'][1]:
            winning_team = 1

        # Add winning points to agent rewards
        if winning_team is not None:
            for agent_idx in range(self.N_AGENTS):
                if self.AGENT_TEAMS[agent_idx] == winning_team:
                    rewards[agent_idx] += score_margin * self.WIN_MARGIN_SCALAR
                elif penalise_losing_team:
                    rewards[agent_idx] -= score_margin * self.LOSS_MARGIN_SCALAR

        return rewards

    def get_global_rewards(self, rewards):
        """
        Get global rewards.
        """
        team_rewards = {0:0, 1:0}
        for i in range(self.N_AGENTS):
            # Sum team rewards
            team_rewards[self.AGENT_TEAMS[i]] += rewards[i]

            # Adjust opponent points for flag capture
            if rewards[i] == (self.REWARD_CAPTURE - 1):
                team_rewards[1 - self.AGENT_TEAMS[i]] -= (rewards[i] - 1)

        return [team_rewards[self.AGENT_TEAMS[i]] for i in range(self.N_AGENTS)]
    
    def get_adjusted_rewards(self, rewards):
        """
        Adjust rewards to opponent flag capture.
        """

        # Now adjust the rewards for each agent by subtracting opponent rewards
        for i in range(self.N_AGENTS):
            rewards[i] -= self._flag_capture_team_current_move[1 - self.AGENT_TEAMS[i]] * self.REWARD_CAPTURE * self.OPP_FLAG_CAPTURE_PUNISHMENT_SCALAR
            
        return rewards
            
    def get_reversed_action(self, action):
        """
        Get reversed action for opponent.
        """
           
        return self.REVERSED_ACTION_MAP[self.FLIP_AXIS][action]
    
    def standardise_state(self, agent_idx, reverse_grid=False):
        """
        Standardises the environment state:
            1) Tiles are standardised to look the same for each team.
            2) Ego state is generated if chosen.
            3) Tile values are scaled between 0 and 1.
        """

        # Copy game grid
        grid = self.grid.copy()

        # Agent own team and opposition
        for agent_idx_, team in self.AGENT_TEAMS.items():
            std_tile = (self.STD_AGENT_TYPE_TILES_OWN if team == self.AGENT_TEAMS[agent_idx]
                        else self.STD_AGENT_TYPE_TILES_OPP)
            agent_tile = self.AGENT_TILE_MAP[agent_idx_]
            grid[np.equal(self.grid, agent_tile)] = std_tile[self.AGENT_TYPES[agent_idx_]]

        grid[np.equal(self.grid, self.FLAG_TILE_MAP[1 - self.AGENT_TEAMS[agent_idx]])] = self.STD_OPP_FLAG_TILE
        grid[np.equal(self.grid, self.FLAG_TILE_MAP[self.AGENT_TEAMS[agent_idx]])] = self.STD_OWN_FLAG_TILE

        agent_x, agent_y = self.agent_positions[agent_idx]
        multi_channel_grid = np.zeros((len(self.TILES_USED) + 1, self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        multi_channel_grid[0, agent_x, agent_y] = 1

        for i, tile_value in enumerate(self.TILES_USED):
            multi_channel_grid[i + 1, :, :] = (grid == tile_value)

        if reverse_grid:
            if self.FLIP_AXIS != 2:
                multi_channel_grid = np.array([np.flip(multi_channel_grid[i, :, :], self.FLIP_AXIS) for i in range(multi_channel_grid.shape[0])])
            else:
                multi_channel_grid = np.array([np.rot90(multi_channel_grid[i, :, :].T, 2) for i in range(multi_channel_grid.shape[0])])

        return np.expand_dims(multi_channel_grid, axis=0)

    def get_env_dims(self):
        """
        Get local and global dims.
        """

        local_grid_dims = (len(self.TILES_USED)+1, self.GRID_SIZE, self.GRID_SIZE)
        global_grid_dims = (len(self.TILES_USED), self.GRID_SIZE, self.GRID_SIZE)

        # Hardcoded as 6, as there are :
        #   - 1 bit of information per agent for agent teams, flag indicator and hitpoints % 
        #   - 3n bits of information for agent types (3 bits per agent, one bit per class)
        local_metadata_dims = (self.N_AGENTS*2 + 6, )
        global_metadata_dims = (self.N_AGENTS*6 + self.N_AGENTS*self.ACTION_SPACE + 3, )

        return local_grid_dims, global_grid_dims, local_metadata_dims, global_metadata_dims
    
    def get_env_metadata(self, agent_idx):
        """
        Return local metadata from the environment.
        """

        EPSILON = 1
        
        # Get game state data
        game_percent_complete = self.env_step_count / self.GAME_STEPS
        team_margin_pct = (self.metrics['team_flag_captures'][self.AGENT_TEAMS[agent_idx]] + EPSILON) / (self.metrics['team_flag_captures'][1 - self.AGENT_TEAMS[agent_idx]] + EPSILON)

        # Get agent hitpoints
        agent_hp = np.zeros(self.N_AGENTS, dtype=np.uint8)
        for i, (k, v) in enumerate(self.AGENT_TYPES.items()):
            agent_hp[i] = self.agent_hp[v] / self.AGENT_TYPE_HP[self.AGENT_TYPES[i]]

        # Populate array
        metadata_state = np.zeros(6 + self.N_AGENTS * 2, dtype=np.float16)
        metadata_state[0] = game_percent_complete
        metadata_state[1] = team_margin_pct
        metadata_state[2 + self.AGENT_TYPES[agent_idx]] = 1.0

        # Agent specific data
        metadata_state[6] = agent_hp[agent_idx]
        metadata_state[7] = self.has_flag[agent_idx]

        idx = 8
        # Get team data
        for teammate_idx in self.OPPONENTS[1-self.AGENT_TEAMS[agent_idx]]:
            if teammate_idx != agent_idx:
                metadata_state[idx] = agent_hp[teammate_idx]
                idx += 1
                metadata_state[idx] = self.has_flag[teammate_idx]
                idx += 1

        # Get opponent data
        for opponent_idx in self.OPPONENTS[self.AGENT_TEAMS[agent_idx]]:
                metadata_state[idx] = agent_hp[opponent_idx]
                idx += 1
                metadata_state[idx] = self.has_flag[opponent_idx]
                idx += 1

        return np.expand_dims(metadata_state, axis=0)

    def render(self, sleep_time: float=0.2, ego_state_agent=None):
        """
        Renders the current game grid using matplotlib.

        Adapted from Bath RL online MSc unit code.
        """
        # Turn interactive mode on.
        plt.ion()
        fig = plt.figure(num="env_render")
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment plot
        if ego_state_agent is None:
            env_plot = np.copy(self.grid)
        else:
            env_plot = self.standardise_state(ego_state_agent, use_ego_state=True)

        # Make a 3d numpy array that has a color channel dimension   
        data_3d = np.ndarray(shape=(env_plot.shape[0], env_plot.shape[1], 3), dtype=int)
        for i in range(0, env_plot.shape[0]):
            for j in range(0, env_plot.shape[1]):
                data_3d[i][j] = self.COLOUR_MAP[env_plot[i][j]]

        # Plot the gridworld
        ax.imshow(data_3d)

        # Set up axes
        ax.grid(which='major', axis='both', linestyle='-', color='0.4', linewidth=2, zorder=1)
        ax.set_xticks(np.arange(-0.5, env_plot.shape[1] , 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, env_plot.shape[0], 1))
        ax.set_yticklabels([])

        plt.show()

        # Sleep if desired
        if (sleep_time > 0) :
            time.sleep(sleep_time)

    def render_image(self, frame_path=None, plot_image=False):
        """
        Renders the current game grid using matplotlib.

        Adapted from Bath RL online MSc unit code.
        """

        # List to store paths to the saved image frames
        env_plot = np.copy(self.grid)

        # Create a figure to contain the subplots
        fig, ax = plt.subplots(self.GRID_SIZE, 
                               self.GRID_SIZE, 
                               figsize=(5, 5), 
                               gridspec_kw={'wspace':0.04, 'hspace':0.04},
                               squeeze=True)
        
        teams_with_flag = {0:0, 1:0}
        agent_positions_reversed = {v: k for k, v in self.agent_positions.items()}

        for agent_idx, flag in enumerate(self.has_flag):
            if flag == 1:
                teams_with_flag[self.AGENT_TEAMS[agent_idx]] = 1

        # Iterate over the rows of the array
        for i in range(self.GRID_SIZE):
            # Iterate over the columns of the array
            for j in range(self.GRID_SIZE):
                # Get the corresponding number
                img_idx = env_plot[i, j]

                # Red agent has flag
                if ((i, j) == self.FLAG_POSITIONS[0] and teams_with_flag[1] == 1):
                    img_idx = 112
                
                # Blue agent has flag
                if ((i, j) == self.FLAG_POSITIONS[1] and teams_with_flag[0] == 1):
                    img_idx = 113

                # Render agent with flag
                if (i, j) in agent_positions_reversed:
                    agent_idx = agent_positions_reversed[(i, j)]
                    if self.has_flag[agent_idx] == 1:
                        img_idx += 100

                # Display the image in the corresponding subplot
                ax[i, j].imshow(self.IMAGE_MAP[img_idx])
                ax[i, j].axis('off')  # to hide the axis

        if plot_image:
            plt.show()

        # Save the figure to a file
        if frame_path is not None:
            fig.savefig(frame_path, dpi=300)
            # Close the figure to free up memory
            plt.close(fig)

    def play(self, 
             player=0, 
             agents=None, 
             use_ego_state=False, 
             device='cpu',
             render_ego_state=False):
        """
        Play the environment manually.
        """

        self.reset()

        move_counter = 0
        total_score = 0

        ACTIONS_MAP = {
                        'w':0,
                        's':1,
                        'd':2,
                        'a':3,
                        'x':4,
                        't':5,
                        'g':6,
                        'h':7,
                        'f':8,
                    }

        raw_action = None

        # Get agents if provided
        if agents is not None:
            agent_t1 = agents[0]
            agent_t2 = agents[1]

        if render_ego_state:
            self.render(ego_state_agent=player)
        else:
            self.render()

        env_dims = self.ENV_DIMS

        # Start main loop
        while True:  
            print(f"Move {move_counter}")
            
            if move_counter > 1:
                print(f"reward: {rewards[0]}")
                print(f"flag captures: {self.metrics['agent_flag_captures'][player]}")
                print(f"hp: {self.agent_hp[player]}")
                print(f"done: {done}")

            while raw_action not in ['w', 's', 'a', 'd', 'q', 'e', 'z', 'x', 't', 'g', 'h', 'f', 'p', 'r', 'y', 'v', 'b']:
                raw_action = input("Enter an action")

            if raw_action == 'p':
                print(f"Game exited")
                break

            # Initialise random actions
            actions = np.random.randint(8, size=self.N_AGENTS)

            # If agents are supplied, get agent actions
            if agents is not None:
                for agent_idx in np.arange(self.N_AGENTS):
                    metadata_state = torch.from_numpy(self.get_env_metadata(agent_idx)).float().to(device)
                    
                    grid_state_ = self.standardise_state(agent_idx, use_ego_state=use_ego_state).reshape(*env_dims) + ut.add_noise(env_dims)
                    grid_state = torch.from_numpy(grid_state_).float().to(device)

                    if self.AGENT_TEAMS[agent_idx]==0:
                        actions[agent_idx] = agent_t1.choose_action(grid_state, metadata_state)
                    else:
                        actions[agent_idx] = agent_t2.choose_action(grid_state, metadata_state)

            # Insert player action
            actions[player] = int(ACTIONS_MAP[raw_action])

            print(f"action chosen: {ACTIONS_MAP[raw_action]}")

            # Step the environment
            _, rewards, done = self.step(actions)

            total_score += rewards[0]
            move_counter += 1

            raw_action = None
            if render_ego_state:
                self.render(ego_state_agent=player)
            else:
                self.render()

            if done:
                print(f"You win!, total score {total_score}")
                break