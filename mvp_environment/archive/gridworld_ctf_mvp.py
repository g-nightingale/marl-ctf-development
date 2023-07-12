import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import utils as ut
from collections import defaultdict
from functools import partial

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
                RANDOMISE_FLAG_POSITIONS=False,
                GRID_SIZE=10,
                ENABLE_PICKUPS=False,
                ENABLE_OBSTACLES=False,
                DROP_FLAG_WHEN_NO_HP=False,
                HOME_FLAG_CAPTURE=True,
                USE_EASY_CAPTURE=True,
                USE_ADJUSTED_REWARDS=False,
                MAX_BLOCK_TILE_PCT=0.2,
                MAX_HEALTH_PICKUP_TILES=1,
                LOG_METRICS=True
                ):


        #----------------------------------------------------------------------------------
        # General Config
        #----------------------------------------------------------------------------------
        self.AGENT_CONFIG = AGENT_CONFIG
        self.SCENARIO = SCENARIO
        self.GAME_STEPS = GAME_STEPS
        self.RANDOMISE_FLAG_POSITIONS = RANDOMISE_FLAG_POSITIONS
        self.GRID_SIZE = GRID_SIZE
        self.ENV_DIMS = (1, 1, GRID_SIZE, GRID_SIZE)
        self.ENABLE_PICKUPS = ENABLE_PICKUPS
        self.ENABLE_OBSTACLES = ENABLE_OBSTACLES
        self.DROP_FLAG_WHEN_NO_HP = DROP_FLAG_WHEN_NO_HP
        self.HOME_FLAG_CAPTURE = HOME_FLAG_CAPTURE
        self.USE_EASY_CAPTURE = USE_EASY_CAPTURE
        self.MAX_BLOCK_TILE_PCT = MAX_BLOCK_TILE_PCT
        self.N_AGENTS = len(self.AGENT_CONFIG)
        self.LOG_METRICS = LOG_METRICS
        self.ACTION_SPACE = 8
        self.METADATA_VECTOR_LEN = 16
        
        # Rewards
        self.REWARD_WIN = 10
        self.REWARD_CAPTURE = 1
        self.REWARD_STEP = 0
        self.WINNING_POINTS = np.inf
        self.USE_ADJUSTED_REWARDS = USE_ADJUSTED_REWARDS

        # Block counts for constructor agents
        self.BLOCK_PICKUP_VALUE = 1
        self.HEALTH_PICKUP_VALUE = 1
        self.MAX_HEALTH_PICKUP_TILES = MAX_HEALTH_PICKUP_TILES

        # Position of flags for each team
        self.FLAG_POSITIONS = {
            0: (1, 1),
            1: (self.GRID_SIZE-2, self.GRID_SIZE-2)
        }

        # Capture positions for each team
        self.CAPTURE_POSITIONS = {
            0: (1, 1),
            1: (self.GRID_SIZE-2, self.GRID_SIZE-2)
        }

        self._arr = np.arange(self.N_AGENTS)

        #----------------------------------------------------------------------------------
        # Agent Config
        #----------------------------------------------------------------------------------

        # Starting positions for each agent
        self.AGENT_STARTING_POSITIONS = {
            0: (0, 2),
            1: (GRID_SIZE-1, GRID_SIZE-3),
            2: (2, 0),
            3: (GRID_SIZE-3, GRID_SIZE-1),
            4: (2, 2),
            5: (GRID_SIZE-3, GRID_SIZE-3),
        }

        # Teams that agent's belong to: 0 = team 1, 1 = team 2
        self.AGENT_TEAMS = {k:self.AGENT_CONFIG[k]['team'] for k in self.AGENT_CONFIG.keys()}

        # Agent types: 0 = flag carrier, 1 = tagger, 2 = constructor
        self.AGENT_TYPES = {k:self.AGENT_CONFIG[k]['type'] for k in self.AGENT_CONFIG.keys()}

        # Agent hitpoints
        self.AGENT_TYPE_HP = {
            0: 5,
            1: 4,
            2: 2
        }

        # Damage dealt by each agent type
        self.AGENT_TYPE_DAMAGE = {
            0: 0,
            1: 3,
            2: 0
        }

        # Probability of a successful tag for a tagging agent
        self.TAG_PROBABILITY = 1.0

        # Agent flag capture types - types that can capture the flag
        self.AGENT_FLAG_CAPTURE_TYPES = [0] #[0, 1, 2]

        self.MAX_AGENT_HEALTH = 5
        self.MAX_AGENT_BLOCKS = 5

        #----------------------------------------------------------------------------------
        # Tile definitions
        #----------------------------------------------------------------------------------
        # Open and block tiles
        self.OPEN_TILE = 0
        self.BLOCK_TILE = 1
        self.DESTRUCTIBLE_TILE1 = 2
        self.DESTRUCTIBLE_TILE2 = 3
        self.DESTRUCTIBLE_TILE3 = 4

        # Tile values for each agent type by team
        self.AGENT_TYPE_TILE_MAP = {
            0: {0:5, 1:8},
            1: {0:6, 1:9},
            2: {0:7, 1:10},
        }

        # Derive AGENT_TILE_MAP from AGENT_TYPE_TILE_MAP and AGENT_CONFIG
        self.AGENT_TILE_MAP = {k:self.AGENT_TYPE_TILE_MAP[AGENT_CONFIG[k]['type']][AGENT_CONFIG[k]['team']] for k in AGENT_CONFIG.keys()}

        # Tile values for flags
        self.FLAG_TILE_MAP = {
            0: 11,
            1: 12
        }

        # Block pickup tile
        self.HEALTH_PICKUP_TILE = 13

        self.MAX_TILE_VALUE = 13

        # Tiles used in environment
        self.TILES_USED = self.get_tiles_used()

        # Standardised tiles
        self.STD_BLOCK_TILE = 1
        self.STD_DESTRUCTIBLE_TILE1 = 2
        self.STD_DESTRUCTIBLE_TILE2 = 3
        self.STD_DESTRUCTIBLE_TILE3 = 4

        self.STD_AGENT_TYPE_TILES_OWN = {
            0: 5,
            1: 6,
            2: 7,
        }
        self.STD_AGENT_TYPE_TILES_OPP = {
            0: 8,
            1: 9,
            2: 10,
        }

        self.STD_OWN_FLAG_TILE = 11
        self.STD_OPP_FLAG_TILE = 12
        self.STD_HEALTH_PICKUP_TILE = 13

        # Colour map for rendering environment. See: https://www.rapidtables.com/web/color/RGB_Color.html
        self.COLOUR_MAP = {
                    0: np.array([224, 224, 224]), # light grey
                    1: np.array([0, 0, 0]), # black
                    2: np.array([32, 32, 32]), # black
                    3: np.array([64, 64, 64]), # black
                    4: np.array([128, 128, 128]), # black
                    5: np.array([102, 178, 255]), # blue 
                    6: np.array([0, 128, 255]), # blue
                    7: np.array([153, 204, 255]), # blue
                    8: np.array([255, 102, 102]), # red
                    9: np.array([255, 51, 51]), # red
                    10: np.array([255, 153, 153]), # red
                    11: np.array([0, 0, 153]), # dark blue
                    12: np.array([153, 0, 0]), # dark red
                    13: np.array([153, 51, 255]) # purple
        }  

        # Override standardisation
        self.STANDARDISATION_OVERRIDE = False

        # Reset the environment
        self.reset()

    def load_scenario(self, scenario):
        """
        Load a fixed gameplay scenario.
        """
        # Load attributes
        self.SCENARIO_NAME = scenario['SCENARIO_NAME']
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

        agents_per_team = int(self.N_AGENTS/2)
        self.OPPONENTS = {
            0: [k for k, v in self.AGENT_TEAMS.items() if v==1][:agents_per_team],
            1: [k for k, v in self.AGENT_TEAMS.items() if v==0][:agents_per_team],
        }
        
        # Generate the map
        if self.SCENARIO is None:
            self.generate_map()
        else:
            self.load_scenario(self.SCENARIO)

        # Get agent positions
        self.agent_positions = {k:v for k, v in self.AGENT_STARTING_POSITIONS.items() if k < self.N_AGENTS}

        # Game metadata
        self.has_flag = np.zeros(self.N_AGENTS, dtype=np.uint8)
        self.agent_teams_np = np.array([v for v in self.AGENT_TEAMS.values()])
        
        self.agent_types_np = np.array([0]*len(self.AGENT_TYPES)*3)
        for i in range(len(self.AGENT_TYPES)):
            self.agent_types_np[i*3+[x for x in self.AGENT_TYPES.values()][i]] = 1

        # Agent hitpoints
        self.agent_hp = {kv[0]:self.AGENT_TYPE_HP[kv[1]] for kv in self.AGENT_TYPES.items()}

        # Agent block inventory
        self.block_inventory = {kv[0]:0 if kv[1]==2 else 0 for kv in self.AGENT_TYPES.items()}

        # Spawn block power-ups
        if self.ENABLE_PICKUPS:
            self.spawn_health_tiles()

        # Internal var to track flag capture in current move
        self._flag_capture_current_move = False

        # Metrics dictionary
        self.metrics = {
            # Team level metrics
            "team_wins": {0:0, 1:0},
            "team_tag_count": {0:0, 1:0},
            "team_respawn_tag_count": {0:0, 1:0},
            "team_flag_pickups": {0:0, 1:0},
            "team_flag_captures": {0:0, 1:0},
            "team_blocks_laid": {0:0, 1:0},
            "team_blocks_mined": {0:0, 1:0},
            "team_steps_defending_zone": {0:0, 1:0},
            "team_steps_attacking_zone": {0:0, 1:0},
            "team_health_pickups": {0:0, 1:0},
            # Agent type metrics
            "agent_type_tag_count": defaultdict(lambda: defaultdict(int)),
            "agent_type_respawn_tag_count": defaultdict(lambda: defaultdict(int)),
            "agent_type_flag_pickups": defaultdict(lambda: defaultdict(int)),
            "agent_type_flag_captures": defaultdict(lambda: defaultdict(int)),
            "agent_type_blocks_laid": defaultdict(lambda: defaultdict(int)),
            "agent_type_blocks_mined": defaultdict(lambda: defaultdict(int)),
            "agent_type_steps_defending_zone": defaultdict(lambda: defaultdict(int)),
            "agent_type_steps_attacking_zone": defaultdict(lambda: defaultdict(int)),
            "agent_type_health_pickups": defaultdict(lambda: defaultdict(int)),
            # Agent level metrics
            "agent_tag_count": defaultdict(int),
            "agent_respawn_tag_count": defaultdict(int),
            "agent_flag_pickups": defaultdict(int),
            "agent_flag_captures": defaultdict(int),
            "agent_blocks_laid": defaultdict(int),
            "agent_blocks_mined": defaultdict(int),
            "agent_steps_defending_zone": defaultdict(int),
            "agent_steps_attacking_zone": defaultdict(int),
            "agent_health_pickups": defaultdict(int),
            "agent_visitation_maps": defaultdict(lambda: np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)),
        }

        # Update the visitation maps with starting positions
        self.update_visitation_map()

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
        
        # Init tile list
        tiles_list = []

        # Add block tile if obstacles are enabled
        if self.ENABLE_OBSTACLES:
            tiles_list.append(self.BLOCK_TILE)

        # Add agent type tiles
        tiles_list.extend(sorted(set(self.AGENT_TILE_MAP.values())))

        # Add destructible tiles if the miner agent is present
        if 2 in set(self.AGENT_TYPES.values()):
            tiles_list.extend(
                [
                    self.DESTRUCTIBLE_TILE1,
                    self.DESTRUCTIBLE_TILE2,
                    self.DESTRUCTIBLE_TILE3
                ]
            )      

        # Flag tiles will always be present
        tiles_list.extend(set(self.FLAG_TILE_MAP.values()))

        # Add health pickup if pickups are enabled
        if self.ENABLE_PICKUPS:
                tiles_list.extend(self.HEALTH_PICKUP_TILE)

        return tiles_list

    def generate_map(self):
        """
        Randomly generate a map.
        """

        # Initialise grid
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        # Decide flag positions zones
        if self.RANDOMISE_FLAG_POSITIONS:
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
        if self.RANDOMISE_FLAG_POSITIONS:
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
        else:
            for agent_idx in range(self.N_AGENTS):
                grid[self.AGENT_STARTING_POSITIONS[agent_idx]] = self.AGENT_TILE_MAP[agent_idx]

        self.grid = grid

    def movement_handler(self, curr_pos, new_pos, agent_idx):
        """
        Handles agent movements and block tile damage.
        """

        # Get flag positions - these locations are sacred!
        flag_positions = [v for v in self.FLAG_POSITIONS.values()]

        #----------------------------------------------------------------------------------
        # Movement into a free tile
        #----------------------------------------------------------------------------------
        if self.grid[new_pos] == self.OPEN_TILE:
            self.grid[curr_pos] = 0
            self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
            curr_pos = new_pos

            # Easy capture movements are not mutually exclusive! 
            if self.USE_EASY_CAPTURE is True:
                # Move into opponent flag
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

        #----------------------------------------------------------------------------------
        # Movement into opponent flag
        #----------------------------------------------------------------------------------
        elif self.USE_EASY_CAPTURE is False \
          and self.grid[new_pos] == self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]] \
          and self.AGENT_TYPES[agent_idx] in self.AGENT_FLAG_CAPTURE_TYPES:
            self.has_flag[agent_idx] = 1
            self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.BLOCK_TILE

        #----------------------------------------------------------------------------------
        # Flag capture
        #----------------------------------------------------------------------------------
        elif self.USE_EASY_CAPTURE is False \
          and new_pos == self.FLAG_POSITIONS[self.AGENT_TEAMS[agent_idx]] \
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
                if self.metrics['team_flag_captures'][self.AGENT_TEAMS[agent_idx]] == self.WINNING_POINTS:
                    self.done = True
                # Update indicator
                self._flag_capture_current_move = True

        #----------------------------------------------------------------------------------
        # Check for tag
        #----------------------------------------------------------------------------------
        elif new_pos in [kv[1] for kv in self.agent_positions.items() if kv[0] in self.OPPONENTS[self.AGENT_TEAMS[agent_idx]]] \
          and self.AGENT_TYPE_DAMAGE[self.AGENT_TYPES[agent_idx]]>0: 
            if np.random.rand() < self.TAG_PROBABILITY:
                reverse_map =  {v: k for k, v in self.agent_positions.items()}
                opp_agent_idx = reverse_map[new_pos]
                # Reduce opponent hit points by the amount of damage incurred by the agent type
                self.agent_hp[opp_agent_idx] -= self.AGENT_TYPE_DAMAGE[self.AGENT_TYPES[agent_idx]]
                # If hitpoints are zero, return flag and respawn opponent agent
                # Update metrics
                self.metrics['team_tag_count'][self.AGENT_TEAMS[agent_idx]]  += 1
                self.metrics['agent_type_tag_count'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_tag_count'][agent_idx]  += 1
                if self.agent_hp[opp_agent_idx] <= 0:
                    # Respawn opponent agent (also handles flag return)
                    self.respawn(opp_agent_idx)
                    self.metrics['team_respawn_tag_count'][self.AGENT_TEAMS[agent_idx]] += 1
                    self.metrics['agent_type_respawn_tag_count'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                    self.metrics['agent_respawn_tag_count'][agent_idx]  += 1
                    
        #----------------------------------------------------------------------------------
        # Check for pickup
        #----------------------------------------------------------------------------------
        # elif self.grid[new_pos] == self.HEALTH_PICKUP_TILE:
        #     self.grid[curr_pos] = 0
        #     self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
        #     if self.agent_hp[agent_idx] < self.MAX_AGENT_HEALTH:
        #         self.agent_hp[agent_idx] += self.HEALTH_PICKUP_VALUE
        #     # Update metrics
        #     self.metrics['team_health_pickups'][self.AGENT_TEAMS[agent_idx]] += 1
        #     self.metrics['agent_type_health_pickups'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
        #     self.metrics['agent_health_pickups'][agent_idx] += 1
        #     curr_pos = new_pos

        #----------------------------------------------------------------------------------
        # Block mining
        #----------------------------------------------------------------------------------
        elif new_pos not in flag_positions and self.AGENT_TYPES[agent_idx]==2:
            if self.grid[new_pos] == self.DESTRUCTIBLE_TILE1:
                self.grid[new_pos] = self.DESTRUCTIBLE_TILE2
            elif self.grid[new_pos] == self.DESTRUCTIBLE_TILE2:
                self.grid[new_pos] = self.DESTRUCTIBLE_TILE3
            elif self.grid[new_pos] == self.DESTRUCTIBLE_TILE3:
                self.grid[new_pos] = self.OPEN_TILE
                if self.block_inventory[agent_idx] < self.MAX_AGENT_BLOCKS:
                    self.block_inventory[agent_idx] += self.BLOCK_PICKUP_VALUE
                # Update metrics
                self.metrics['team_blocks_mined'][self.AGENT_TEAMS[agent_idx]] += 1
                self.metrics['agent_type_blocks_mined'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_blocks_mined'][agent_idx] += 1

        return curr_pos

    def add_block(self, pos, agent_idx):
        """
        Add a block to the environment for the miner agent class.
        """
        self.grid[pos] = self.DESTRUCTIBLE_TILE1
        self.block_inventory[agent_idx] -= 1
        # Update metrics
        self.metrics['team_blocks_laid'][self.AGENT_TEAMS[agent_idx]] += 1
        self.metrics['agent_type_blocks_laid'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
        self.metrics['agent_blocks_laid'][agent_idx] += 1

    def get_reversed_action(self, action):
        """
        Get reversed action for opponent.
        """

        if action % 2 == 0:
            reversed_action = action + 1
        else:
            reversed_action = action - 1
           
        return reversed_action
    
    def act(self, agent_idx, action) -> None:
        """
        Take agent actions and update grid.
        """

        # Init reward
        reward = 0

        # Get the current position of the agent
        curr_pos = self.agent_positions[agent_idx]

        # Standard movements
        up_pos = (curr_pos[0] - 1, curr_pos[1])
        down_pos = (curr_pos[0] + 1, curr_pos[1])
        right_pos = (curr_pos[0], curr_pos[1] + 1)
        left_pos = (curr_pos[0], curr_pos[1] - 1)

        # Diagonal movements
        up_left_pos = (curr_pos[0] - 1, curr_pos[1] - 1)
        up_right_pos = (curr_pos[0] - 1, curr_pos[1] + 1)
        down_left_pos = (curr_pos[0] + 1, curr_pos[1] - 1)
        down_right_pos = (curr_pos[0] + 1, curr_pos[1] + 1)

        #----------------------------------------------------------------------------------
        # Movement actions for all agents
        #----------------------------------------------------------------------------------
        # Move up
        if action == 0:
            if curr_pos[0] > 0:
                curr_pos = self.movement_handler(curr_pos, up_pos, agent_idx)
        # Move down
        elif action == 1:
            if curr_pos[0] < (self.GRID_SIZE-1):
                curr_pos = self.movement_handler(curr_pos, down_pos, agent_idx)
        # Move right
        elif action == 2:
            if curr_pos[1] < (self.GRID_SIZE-1):
                curr_pos = self.movement_handler(curr_pos, right_pos, agent_idx)
        # Move left
        elif action == 3:
            if curr_pos[1] > 0:
                curr_pos = self.movement_handler(curr_pos, left_pos, agent_idx)
        # Move up left
        if action == 4:
            if curr_pos[0] > 0 and curr_pos[1] > 0:
                curr_pos = self.movement_handler(curr_pos, up_left_pos, agent_idx)
        # Move up right
        elif action == 5:
            if curr_pos[0] > 0 and curr_pos[1] < (self.GRID_SIZE-1):
                curr_pos = self.movement_handler(curr_pos, up_right_pos, agent_idx)
        # Move down left
        elif action == 6:
            if curr_pos[0] < (self.GRID_SIZE-1) and curr_pos[1] > 0:
                curr_pos = self.movement_handler(curr_pos, down_left_pos, agent_idx)
        # Move down right
        elif action == 7:
            if curr_pos[0] < (self.GRID_SIZE-1) and curr_pos[1] < (self.GRID_SIZE-1):
                curr_pos = self.movement_handler(curr_pos, down_right_pos, agent_idx)          

        #----------------------------------------------------------------------------------
        # Actions specific to builder agent
        #----------------------------------------------------------------------------------
        # Add block up top
        elif action == 8 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] > 0 and self.grid[up_pos] == self.OPEN_TILE:
                self.add_block(up_pos, agent_idx)
        # Add block on bottom
        elif action == 9 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] < (self.GRID_SIZE-1) and self.grid[down_pos] == self.OPEN_TILE:
                self.add_block(down_pos, agent_idx)
        # Add block to right
        elif action == 10 and self.block_inventory[agent_idx] > 0:
            if curr_pos[1] < (self.GRID_SIZE-1) and self.grid[right_pos] == self.OPEN_TILE:
                self.add_block(right_pos, agent_idx)
        # Add block to left
        elif action == 11 and self.block_inventory[agent_idx] > 0:
            if curr_pos[1] > 0 and self.grid[left_pos] == self.OPEN_TILE:
                self.add_block(left_pos, agent_idx)

        # Add block up top left
        elif action == 12 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] > 0 and curr_pos[1] > 0 and self.grid[up_left_pos] == self.OPEN_TILE:
                self.add_block(up_left_pos, agent_idx)
        # Add block on top right
        elif action == 13 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] > 0 and curr_pos[1] < (self.GRID_SIZE-1) and self.grid[up_right_pos] == self.OPEN_TILE:
                self.add_block(up_right_pos, agent_idx)
        # Add block to bottom left
        elif action == 14 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] < (self.GRID_SIZE-1) and curr_pos[1] > 0 and self.grid[down_left_pos] == self.OPEN_TILE:
                self.add_block(down_left_pos, agent_idx)
        # Add block to bottom right
        elif action == 15 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] < (self.GRID_SIZE-1) and curr_pos[1] < (self.GRID_SIZE-1) and self.grid[down_right_pos] == self.OPEN_TILE:
                self.add_block(down_right_pos, agent_idx)

        # Do nothing -> removed for now
        # elif action == 8:
        #     pass

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

        np.random.shuffle(self._arr)

        return self._arr

    def agent_one_cell_distance_to_object(self, agent_idx, object_tile) -> bool:
        """
        Check if agent is within one cell of a given object.
        Used for checking flag captures and tags.
        """
        
        x, y = self.agent_positions[agent_idx]
        capture = self.grid[max(x-1, 0):x+2, max(y-1, 0):y+2] == object_tile

        return capture.any()

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
    
    def get_tile_pos_within_one_cell(self, agent_idx, object_tile):
        """
        Returns all positions of a specified tile within one cell of an agent.
        """
        x, y = self.agent_positions[agent_idx]
        return np.where(self.grid[max(x-1, 0):x+2, max(y-1, 0):y+2] == object_tile)

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
            if self.DROP_FLAG_WHEN_NO_HP:
                self.grid[old_pos] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]
            else:
                self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]

    def spawn_pickup(self, pickup_tile):
        """
        Spawn a pickup tile on a free tile the map somewhere.
        """

        # Find all the free tiles on the grid
        open_tiles = np.where(self.grid==self.OPEN_TILE)

        # Choose a random spawn location at random
        rnd = np.random.randint(open_tiles[0].shape[0])
        
        # Get spawn location
        pickup_pos = (open_tiles[0][rnd], open_tiles[1][rnd])

        # Update tiles
        self.grid[pickup_pos] = pickup_tile

    def spawn_health_tiles(self):
        """
        Spawn pickup tiles.
        """

        while True:
            if (self.grid==self.OPEN_TILE).sum() > 0 and (self.grid==self.HEALTH_PICKUP_TILE).sum() < self.MAX_HEALTH_PICKUP_TILES:
                self.spawn_pickup(self.HEALTH_PICKUP_TILE)
            else:
                break

    def step(self, actions):
        """
        Take a step in the environment.

        Takes in a vector or actions, where each position in the vector represents an agent,
        and the value represents the action to take.
        """

        self.env_step_count += 1

        rewards = [0] * self.N_AGENTS
        for agent_idx in self.dice_roll():

            # Get the agent team
            agent_team = self.AGENT_TEAMS[agent_idx]

            # Get the agent action
            action = actions[agent_idx]

            # Move the agent and get reward
            rewards[agent_idx] = self.act(agent_idx, action)

            #----------------------------------------------------------------------------------
            # Logging extra metrics
            #----------------------------------------------------------------------------------
            distance_to_own_flag = self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[agent_team])
            distance_to_opp_flag = self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[1 - agent_team])

            if distance_to_own_flag <= 2:
                self.metrics['team_steps_defending_zone'][self.AGENT_TEAMS[agent_idx]] += 1
                self.metrics['agent_type_steps_defending_zone'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_steps_defending_zone'][agent_idx] += 1
            if distance_to_opp_flag <= 2:
                self.metrics['team_steps_attacking_zone'][self.AGENT_TEAMS[agent_idx]] += 1
                self.metrics['agent_type_steps_attacking_zone'][self.AGENT_TEAMS[agent_idx]][self.AGENT_TYPES[agent_idx]] += 1
                self.metrics['agent_steps_attacking_zone'][agent_idx] += 1
            
        if self.USE_ADJUSTED_REWARDS:
            rewards = self.get_adjusted_rewards(rewards)

        # Update the visitation maps with new positions
        self.update_visitation_map()

        # Spawn new block tiles
        if self.ENABLE_PICKUPS:
            self.spawn_health_tiles()

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
        winning_team = -1
        if self.metrics['team_flag_captures'][0] > self.metrics['team_flag_captures'][1]:
            winning_team = 0
        elif self.metrics['team_flag_captures'][0] < self.metrics['team_flag_captures'][1]:
            winning_team = 1

        # Add winning points to agent rewards
        for agent_idx in range(self.N_AGENTS):
            if self.AGENT_TEAMS[agent_idx] == winning_team:
                rewards[agent_idx] += self.REWARD_WIN
            elif penalise_losing_team:
                rewards[agent_idx] -= self.REWARD_WIN

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
        Adjust rewards.
        """

        # First get the total rewards for each team
        team_rewards = {0:0, 1:0}
        for i in range(self.N_AGENTS):
            # Sum team rewards
            team_rewards[self.AGENT_TEAMS[i]] += rewards[i]

        # Now adjust the rewards for each agent by subtracting opponent rewards
        for i in range(self.N_AGENTS):
            rewards[i] -= team_rewards[1 - self.AGENT_TEAMS[i]]
            
        return rewards
            

    def standardise_state(self, agent_idx, use_ego_state=False, use_multi_channel=True, reverse_grid=False):
        """
        Standardises the environment state:
            1) Tiles are standardised to look the same for each team.
            2) Ego state is generated if chosen.
            3) Tile values are scaled between 0 and 1.
        """

        # Override standardisation and return environment grid
        if self.STANDARDISATION_OVERRIDE:
            return self.grid

        # Copy game grid
        grid = self.grid.copy()

        # Agent own team
        for agent_idx_ in [kv[0] for kv in self.AGENT_TEAMS.items() if kv[1]==self.AGENT_TEAMS[agent_idx] and kv[0]<self.N_AGENTS]:
            grid[self.grid==self.AGENT_TILE_MAP[agent_idx_]] = self.STD_AGENT_TYPE_TILES_OWN[self.AGENT_TYPES[agent_idx_]]
        # Agent opposition
        for agent_idx_ in [kv[0] for kv in self.AGENT_TEAMS.items() if kv[1]!=self.AGENT_TEAMS[agent_idx] and kv[0]<self.N_AGENTS]:
            grid[self.grid==self.AGENT_TILE_MAP[agent_idx_]] = self.STD_AGENT_TYPE_TILES_OPP[self.AGENT_TYPES[agent_idx_]]

        grid[self.grid==self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]] = self.STD_OPP_FLAG_TILE
        grid[self.grid==self.FLAG_TILE_MAP[self.AGENT_TEAMS[agent_idx]]] = self.STD_OWN_FLAG_TILE
        
        if use_multi_channel:
            if use_ego_state:
                agent_x, agent_y = self.agent_positions[agent_idx]
                multi_channel_grid = np.zeros((len(self.TILES_USED)+1, self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
                multi_channel_grid[0, agent_x, agent_y] = 1
                for i, tile_value in enumerate(self.TILES_USED):
                    multi_channel_grid[i+1, :, :] = np.where(grid==tile_value, 1, 0)
            else:
                multi_channel_grid = np.zeros((len(self.TILES_USED), self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
                for i, tile_value in enumerate(self.TILES_USED):
                    multi_channel_grid[i, :, :] = np.where(grid==tile_value, 1, 0)
            
            grid = multi_channel_grid
        else:
            # Need to expand dims even if we're not using multichannel
            grid = np.expand_dims(grid, axis=0)

        if reverse_grid:
            grid = np.array([np.flip(grid[i, :, :]) for i in range(grid.shape[0])])

        return np.expand_dims(grid, axis=0)

    def get_env_dims(self):
        """
        Get local and global dims.
        """

        local_grid_dims = (len(self.TILES_USED)+1, self.GRID_SIZE, self.GRID_SIZE)
        global_grid_dims = (len(self.TILES_USED), self.GRID_SIZE, self.GRID_SIZE)

        # Hardcoded as 6, as there are :
        #   - 1 bit of information per agent for agent teams, flag indicator and hitpoints % 
        #   - 3n bits of information for agent types (3 bits per agent, one bit per class)
        local_metadata_dims = (self.N_AGENTS*2 + 5, )
        global_metadata_dims = (self.N_AGENTS*6 + self.N_AGENTS*self.ACTION_SPACE + 3, )

        return local_grid_dims, global_grid_dims, local_metadata_dims, global_metadata_dims

    def display_grid(self):
        """
        Display the current grid.
        """

        print(self.grid, '\n')

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

    def get_env_metadata_local_old(self):
        """
        Return local metadata from the environment.
        """

        EPSILON = 1e-6
        percent_complete_np = np.array([self.env_step_count / self.GAME_STEPS])

        scores = np.array([v for v in self.metrics['team_flag_captures'].values()])
        teams_max_score_pct_np = (scores + EPSILON) / (max(scores) + EPSILON) 
        agent_hp = np.array([self.agent_hp[kv[1]]/self.AGENT_TYPE_HP[self.AGENT_TYPES[i]] \
            for i, kv in enumerate(self.AGENT_TYPES.items())], dtype=np.uint8)
      
        local_metadata = np.concatenate((
            percent_complete_np,
            teams_max_score_pct_np,
            self.agent_teams_np,
            self.agent_types_np,
            self.has_flag,
            agent_hp), dtype=np.float16)

        return np.expand_dims(local_metadata, axis=0)
    
    def get_env_metadata_local(self, agent_idx):
        """
        Return local metadata from the environment.
        """

        EPSILON = 1e-6

        # Get game state data
        game_percent_complete = self.env_step_count / self.GAME_STEPS
        scores = np.array([v for v in self.metrics['team_flag_captures'].values()])
        teams_max_score_pct_np = (self.metrics['team_flag_captures'][self.AGENT_TEAMS[agent_idx]] + EPSILON) / (max(scores) + EPSILON) 

        # Get agent hitpoints
        agent_hp = np.array([self.agent_hp[kv[1]]/self.AGENT_TYPE_HP[self.AGENT_TYPES[i]] \
            for i, kv in enumerate(self.AGENT_TYPES.items())], dtype=np.uint8)
        
        # Populate array
        metadata_state = np.zeros(5 + self.N_AGENTS*2, dtype=np.float16)
        metadata_state[0] = game_percent_complete
        metadata_state[1] = teams_max_score_pct_np
        metadata_state[2 + self.AGENT_TYPES[agent_idx]] = 1.0

        # Agent specific data
        metadata_state[5] = agent_hp[agent_idx]
        metadata_state[6] = self.has_flag[agent_idx]

        idx = 7
        # Get team data
        for teammate_idx in self.OPPONENTS[1-self.AGENT_TEAMS[agent_idx]]:
            if teammate_idx != agent_idx:
                metadata_state[idx] = agent_hp[teammate_idx]
                idx += 1
                metadata_state[idx] = self.has_flag[agent_idx]
                idx += 1

        # Get opponent data
        for opponent_idx in self.OPPONENTS[self.AGENT_TEAMS[agent_idx]]:
                metadata_state[idx] = agent_hp[opponent_idx]
                idx += 1
                metadata_state[idx] = self.has_flag[opponent_idx]
                idx += 1

        return np.expand_dims(metadata_state, axis=0)
    
    def get_env_metadata_global(self, actions, use_one_hot_actions=True):
        """
        Return global metadata from the environment.
        """

        EPSILON = 1e-6
        percent_complete_np = np.array([self.env_step_count / self.GAME_STEPS])

        scores = np.array([v for v in self.metrics['team_flag_captures'].values()])
        teams_max_score_pct_np = (scores + EPSILON) / (max(scores) + EPSILON) 
        agent_hp = np.array([self.agent_hp[kv[1]]/self.AGENT_TYPE_HP[self.AGENT_TYPES[i]] \
            for i, kv in enumerate(self.AGENT_TYPES.items())], dtype=np.uint8)

        if use_one_hot_actions:
            actions_np = np.zeros(self.N_AGENTS*self.ACTION_SPACE, dtype=np.uint8)
            for i, action in enumerate(actions):
                actions_np[(i*self.ACTION_SPACE)+action] = 1
        else:
            actions_np = np.array(actions, dtype=np.uint8)

        global_metadata = np.concatenate((
            percent_complete_np,
            teams_max_score_pct_np,
            self.agent_teams_np,
            self.agent_types_np,
            self.has_flag,
            agent_hp,
            actions_np), dtype=np.float16) #.reshape(1, -1)

        return np.expand_dims(global_metadata, axis=0)

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
                        'q':4,
                        'e':5,
                        'z':6,
                        'x':7,
                        't':8,
                        'g':9,
                        'h':10,
                        'f':11,
                        'r':12,
                        'y':13,
                        'v':14,
                        'b':15
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
                    metadata_state = torch.from_numpy(self.get_env_metadata(agent_idx)).reshape(1, self.METADATA_VECTOR_LEN).float().to(device)
                    
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