import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import utils as ut
import torch
from collections import defaultdict

class GridworldCtf:
    """
    A GridWorld capture the flag environment.
    
    """

    def __init__(self, 
                 GAME_MODE='static',
                 GRID_LEN=10,
                 N_AGENTS=4,
                 ENABLE_PICKUPS=False,
                 ENABLE_OBSTACLES=False,
                 MAX_BLOCK_TILE_PCT=0.2,
                 MAX_HEALTH_PICKUP_TILES=1,
                 AGENT_CONFIG = {
                    0: {'team':0, 'type':0},
                    1: {'team':1, 'type':0},
                    2: {'team':0, 'type':1},
                    3: {'team':1, 'type':1},
                    4: {'team':0, 'type':2},
                    5: {'team':1, 'type':2}
                }
                 ) -> None:


        #----------------------------------------------------------------------------------
        # General Config
        #----------------------------------------------------------------------------------
        self.GAME_MODE = GAME_MODE # static or random
        self.GRID_LEN = GRID_LEN
        self.ENV_DIMS = (1, 1, GRID_LEN, GRID_LEN)
        self.EGO_GRID_LEN = GRID_LEN*2 - 1
        self.EGO_ENV_DIMS = (1, 1, self.EGO_GRID_LEN, self.EGO_GRID_LEN)
        self.N_AGENTS = N_AGENTS
        self.ENABLE_PICKUPS = ENABLE_PICKUPS
        self.ENABLE_OBSTACLES = ENABLE_OBSTACLES
        self.MAX_BLOCK_TILE_PCT = MAX_BLOCK_TILE_PCT
        self.AGENT_CONFIG = AGENT_CONFIG
        self.METADATA_VECTOR_LEN = 16
        
        # Rewards
        self.REWARD_CAPTURE = 100
        self.REWARD_STEP = -1
        self.WINNING_POINTS = 3

        # Block counts for constructor agents
        self.BLOCK_PICKUP_VALUE = 1
        self.HEALTH_PICKUP_VALUE = 1
        self.MAX_HEALTH_PICKUP_TILES = MAX_HEALTH_PICKUP_TILES

        # Position of flags for each team
        self.FLAG_POSITIONS = {
            0: (1, 1),
            1: (self.GRID_LEN-2, self.GRID_LEN-2)
        }

        # Capture positions for each team
        self.CAPTURE_POSITIONS = {
            0: (1, 1),
            1: (self.GRID_LEN-2, self.GRID_LEN-2)
        }

        self._arr = np.arange(self.N_AGENTS)

        #----------------------------------------------------------------------------------
        # Agent Config
        #----------------------------------------------------------------------------------

        # Starting positions for each agent
        self.AGENT_STARTING_POSITIONS = {
            0: (0, 2),
            1: (GRID_LEN-3, GRID_LEN-1),
            2: (2, 0),
            3: (GRID_LEN-1, GRID_LEN-3),
            4: (2, 2),
            5: (GRID_LEN-3, GRID_LEN-3),
        }

        # Teams that agent's belong to: 0 = team 1, 1 = team 2
        self.AGENT_TEAMS = {k:self.AGENT_CONFIG[k]['team'] for k in self.AGENT_CONFIG.keys()}

        # Agent types: 0 = flag carrier, 1 = tagger, 2 = constructor
        self.AGENT_TYPES = {k:self.AGENT_CONFIG[k]['type'] for k in self.AGENT_CONFIG.keys()}

        # Agent hitpoints
        self.AGENT_TYPE_HP = {
            0: 3,
            1: 1,
            2: 2
        }

        # Probability of a successful tag for a tagging agent
        self.TAG_PROBABILITY = 0.8

        # Agent flag capture types - types that can capture the flag
        self.AGENT_FLAG_CAPTURE_TYPES = [0, 1, 2]

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

        # Reset the environment
        self.reset()

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
        
        if self.GAME_MODE=='static':
            self.grid = np.zeros((self.GRID_LEN, self.GRID_LEN), dtype=np.uint8)
            self.init_objects()
        elif self.GAME_MODE =='random':
            self.generate_map()
        else:
            raise ValueError("Game mode must be set to either 'static' or 'random'")

        self.agent_positions = {k:v for k, v in self.AGENT_STARTING_POSITIONS.items() if k < self.N_AGENTS}

        # Game metadata
        # self.has_flag = np.zeros(self.N_AGENTS, dtype=np.int8)
        # TEMP -> harcoded for 4 agents
        self.has_flag = np.zeros(4, dtype=np.uint8)
        self.agent_types_np = np.array([v for v in self.AGENT_TYPES.values()])
        self.agent_teams_np = np.array([v for v in self.AGENT_TEAMS.values()])

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
            "team_points": {0:0, 1:0},
            "tag_count": {0:0, 1:0},
            "agent_tag_count": defaultdict(int),
            "agent_flag_captures": defaultdict(int),
            "agent_blocks_laid": defaultdict(int),
            "agent_blocks_mined": defaultdict(int),
            "agent_total_distance_to_own_flag": defaultdict(int),
            "agent_total_distance_to_opp_flag": defaultdict(int),
            "agent_health_pickups": defaultdict(int),
            "agent_visitation_maps": defaultdict(np.zeros((self.GRID_LEN, self.GRID_LEN), dtype=np.uint8))
        }


    def init_objects(self):
        """"
        Initialise the grid.
        """

        # Add agents
        for agent_idx in range(self.N_AGENTS):
            self.grid[self.AGENT_STARTING_POSITIONS[agent_idx]] = self.AGENT_TILE_MAP[agent_idx]

        # Add flags
        self.grid[self.FLAG_POSITIONS[0]] = self.FLAG_TILE_MAP[0]
        self.grid[self.FLAG_POSITIONS[1]] = self.FLAG_TILE_MAP[1]

        # Define spawn positions
        self.SPAWN_POSITIONS = self.FLAG_POSITIONS.copy()

    def generate_map(self):
        """
        Randomly generate a map.
        """

        # Initialise grid
        grid = np.zeros((self.GRID_LEN, self.GRID_LEN), dtype=np.uint8)

        # Place obstacles
        if self.ENABLE_OBSTACLES:
            while (grid==self.DESTRUCTIBLE_TILE1).sum() < self.GRID_LEN**2 * self.MAX_BLOCK_TILE_PCT:
                # Get random coordinates
                x = np.random.randint(0, self.GRID_LEN-1)
                y = np.random.randint(0, self.GRID_LEN-1)

                if np.random.rand() < 0.2:
                    if np.random.rand() < 0.5:
                        grid[x, y:y+3] = self.DESTRUCTIBLE_TILE1
                    else:
                        grid[x:x+3, y] = self.DESTRUCTIBLE_TILE1
                elif np.random.rand() < 0.6:
                    if np.random.rand() < 0.5:
                        grid[x, y:y+2] = self.DESTRUCTIBLE_TILE1
                    else:
                        grid[x:x+2, y] = self.DESTRUCTIBLE_TILE1
                else:
                    grid[x:x+2, y:y+2] = self.DESTRUCTIBLE_TILE1

        # Clear flag + spawning zones
        grid[0:3, 0:3] = self.OPEN_TILE
        grid[0:3, self.GRID_LEN-3:] = self.OPEN_TILE
        grid[self.GRID_LEN-3:, 0:3] = self.OPEN_TILE
        grid[self.GRID_LEN-3:, self.GRID_LEN-3:] = self.OPEN_TILE

        # Place flags and agents
        flag_and_spawn_locations = [
            (1, 1), 
            (1, self.GRID_LEN-2), 
            (self.GRID_LEN-2, 1), 
            (self.GRID_LEN-2, self.GRID_LEN-2)
        ]

        combinations = [
            [0, 1, 2, 3], 
            [0, 1, 3, 2], 
            [1, 0, 2, 3], 
            [1, 0, 3, 2], 
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [2, 0, 1, 3],
            [2, 0, 3, 1]
        ]

        combo = np.random.choice(8)

        if np.random.rand() < 0.5:
            t1_flag = flag_and_spawn_locations[combinations[combo][0]]
            t1_spawn = flag_and_spawn_locations[combinations[combo][1]]
            t2_flag = flag_and_spawn_locations[combinations[combo][2]]
            t2_spawn = flag_and_spawn_locations[combinations[combo][3]]
        else:
            t2_flag = flag_and_spawn_locations[combinations[combo][0]]
            t2_spawn = flag_and_spawn_locations[combinations[combo][1]]
            t1_flag = flag_and_spawn_locations[combinations[combo][2]]
            t1_spawn = flag_and_spawn_locations[combinations[combo][3]]

        # Update tiles
        grid[t1_flag] = self.FLAG_TILE_MAP[0]
        grid[t2_flag] = self.FLAG_TILE_MAP[1]

        # Update positions
        self.FLAG_POSITIONS[0] = t1_flag
        self.FLAG_POSITIONS[1] = t2_flag 

        # Update spawn positions
        self.SPAWN_POSITIONS = {
            0: t1_spawn,
            1: t1_spawn
        }

        for agent_idx in range(self.N_AGENTS):
            # Get agent team flag position - we will respawn around there
            if self.AGENT_TEAMS[agent_idx]==0:
                x, y = t1_spawn
            else:
                x, y = t2_spawn

            possible_respawn_offset = np.where(grid[max(x-1, 0):x+2, max(y-1, 0):y+2]==self.OPEN_TILE)

            # Choose a free respawn location at random
            rnd = np.random.randint(possible_respawn_offset[0].shape[0])
            
            # Get respawn position: WARNING - if the flag is at (0, 0) this will break 
            new_pos = (x + possible_respawn_offset[0][rnd] - 1, y + possible_respawn_offset[1][rnd] - 1)

            # Update tiles
            grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]

            self.AGENT_STARTING_POSITIONS[agent_idx] = new_pos

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

        #----------------------------------------------------------------------------------
        # Movement into opponent flag
        #----------------------------------------------------------------------------------
        elif self.grid[new_pos] == self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]] and self.AGENT_TYPES[agent_idx] in self.AGENT_FLAG_CAPTURE_TYPES:
            self.has_flag[agent_idx] = 1
            self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.BLOCK_TILE

        #----------------------------------------------------------------------------------
        # Flag capture
        #----------------------------------------------------------------------------------
        elif new_pos == self.FLAG_POSITIONS[self.AGENT_TEAMS[agent_idx]] and self.has_flag[agent_idx] == 1:
                self.has_flag[agent_idx] = 0
                self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]
                # Update metrics
                self.metrics['team_points'][self.AGENT_TEAMS[agent_idx]] += 1
                self.metrics['agent_flag_captures'][agent_idx] += 1
                # Check for win
                if self.metrics['team_points'][self.AGENT_TEAMS[agent_idx]] == self.WINNING_POINTS:
                    self.done = True
                # Update indicator
                self._flag_capture_current_move = True

        #----------------------------------------------------------------------------------
        # Check for tag
        #----------------------------------------------------------------------------------
        elif new_pos in [kv[1] for kv in self.agent_positions.items() if kv[0] in self.OPPONENTS[self.AGENT_TEAMS[0]]] and self.AGENT_TYPES[agent_idx]==1: 
            if np.random.rand() < self.TAG_PROBABILITY:
                reverse_map =  {v: k for k, v in self.agent_positions.items()}
                opp_agent_idx = reverse_map[new_pos]
                # Reduce opponent hit points
                self.agent_hp[opp_agent_idx] -= 1
                # If hitpoints are zero, respawn opponent agent
                if self.agent_hp[opp_agent_idx] <= 0:
                    self.respawn(opp_agent_idx)
                # Update metrics
                self.metrics['tag_count'][self.AGENT_TEAMS[agent_idx]]  += 1
                self.metrics['agent_tag_count'][agent_idx]  += 1

        #----------------------------------------------------------------------------------
        # Check for pickup
        #----------------------------------------------------------------------------------
        elif self.grid[new_pos] == self.HEALTH_PICKUP_TILE:
            self.grid[curr_pos] = 0
            self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
            if self.agent_hp[agent_idx] < self.MAX_AGENT_HEALTH:
                self.agent_hp[agent_idx] += self.HEALTH_PICKUP_VALUE
            # Update metrics
            self.metrics['agent_health_pickups'][agent_idx] += 1
            curr_pos = new_pos

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
                self.metrics['agent_blocks_mined'][agent_idx] += 1

        return curr_pos

    def add_block(self, pos, agent_idx):
        """
        Add a block to the environment for the miner agent class.
        """
        self.grid[pos] = self.DESTRUCTIBLE_TILE1
        self.block_inventory[agent_idx] -= 1
        self.metrics['agent_blocks_laid'][agent_idx] += 1

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
            if curr_pos[0] < (self.GRID_LEN-1):
                curr_pos = self.movement_handler(curr_pos, down_pos, agent_idx)
        # Move right
        elif action == 2:
            if curr_pos[1] < (self.GRID_LEN-1):
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
            if curr_pos[0] > 0 and curr_pos[1] < (self.GRID_LEN-1):
                curr_pos = self.movement_handler(curr_pos, up_right_pos, agent_idx)
        # Move down left
        elif action == 6:
            if curr_pos[0] < (self.GRID_LEN-1) and curr_pos[1] > 0:
                curr_pos = self.movement_handler(curr_pos, down_left_pos, agent_idx)
        # Move down right
        elif action == 7:
            if curr_pos[0] < (self.GRID_LEN-1) and curr_pos[1] < (self.GRID_LEN-1):
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
            if curr_pos[0] < (self.GRID_LEN-1) and self.grid[down_pos] == self.OPEN_TILE:
                self.add_block(down_pos, agent_idx)
        # Add block to right
        elif action == 10 and self.block_inventory[agent_idx] > 0:
            if curr_pos[1] < (self.GRID_LEN-1) and self.grid[right_pos] == self.OPEN_TILE:
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
            if curr_pos[0] > 0 and curr_pos[1] < (self.GRID_LEN-1) and self.grid[up_right_pos] == self.OPEN_TILE:
                self.add_block(up_right_pos, agent_idx)
        # Add block to bottom left
        elif action == 14 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] < (self.GRID_LEN-1) and curr_pos[1] > 0 and self.grid[down_left_pos] == self.OPEN_TILE:
                self.add_block(down_left_pos, agent_idx)
        # Add block to bottom right
        elif action == 15 and self.block_inventory[agent_idx] > 0:
            if curr_pos[0] < (self.GRID_LEN-1) and curr_pos[1] < (self.GRID_LEN-1) and self.grid[down_right_pos] == self.OPEN_TILE:
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
        new_pos = (x + possible_respawn_offset[0][rnd] - 1, y + possible_respawn_offset[1][rnd] - 1)

        # Update tiles
        self.grid[self.agent_positions[agent_idx]] = self.OPEN_TILE
        self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
        
        # Update agent position
        self.agent_positions[agent_idx] = new_pos

        # Reset agent hp
        self.agent_hp[agent_idx] = self.AGENT_TYPE_HP[self.AGENT_TYPES[agent_idx]]

        # Reset flag if agent has it
        if self.has_flag[agent_idx]==1:
            self.has_flag[agent_idx] = 0
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

    def step(self, actions) -> tuple:
        """
        Take a step in the environment.

        Takes in a vector or actions, where each position in the vector represents an agent,
        and the value represents the action to take.
        """

        self.env_step_count += 1

        rewards = [0, 0, 0, 0]
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
            self.metrics['agent_total_distance_to_own_flag'][agent_idx] += self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[agent_team])
            self.metrics['agent_total_distance_to_opp_flag'][agent_idx] += self.agent_distance_to_xy(agent_idx, self.CAPTURE_POSITIONS[1-agent_team])

        # Spawn new block tiles
        if self.ENABLE_PICKUPS:
            self.spawn_health_tiles()

        return self.grid, rewards, self.done

    def standardise_state(self, agent_idx, use_ego_state=False, scale_tiles=False):
        """
        Standardises the environment state:
            1) Tiles are standardised to look the same for each team.
            2) Ego state is generated if chosen.
            3) Tile values are scaled between 0 and 1.
        """

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
        # grid[self.grid==self.BLOCK_TILE] = self.STD_BLOCK_TILE
        # grid[self.grid==self.DESTRUCTIBLE_TILE1] = self.STD_DESTRUCTIBLE_TILE1
        # grid[self.grid==self.DESTRUCTIBLE_TILE2] = self.STD_DESTRUCTIBLE_TILE2
        # grid[self.grid==self.DESTRUCTIBLE_TILE3] = self.STD_DESTRUCTIBLE_TILE3
        # grid[self.grid==self.HEALTH_PICKUP_TILE] = self.STD_HEALTH_PICKUP_TILE

        if use_ego_state:
            # Init ego grid
            ego_grid = np.ones((self.EGO_GRID_LEN, self.EGO_GRID_LEN), dtype=np.uint8)

            # Get agent position and calculate where to place the grid in the ego grid
            agent_x, agent_y = self.agent_positions[agent_idx]
            start_x = self.GRID_LEN - agent_x - 1
            start_y = self.GRID_LEN - agent_y - 1

            # Place grid inside the ego grid, centered on the agent
            ego_grid[start_x:start_x+self.GRID_LEN, start_y:start_y+self.GRID_LEN] = grid

            grid = ego_grid

        if scale_tiles:
            grid = np.array(grid / self.MAX_TILE_VALUE, dtype=np.float16) 

        return grid


    def display_grid(self) -> None:
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
            env_plot = self.standardise_state(ego_state_agent, use_ego_state=True, scale_tiles=False)

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

    def get_env_metadata(self, agent_idx):
        """
        Return metadata from the environment.
        """

        metadata = np.zeros(self.METADATA_VECTOR_LEN, dtype=np.float16)
        metadata[agent_idx] = 1
        metadata[4:8] = self.agent_teams_np[:4] # TEMP -> harcoded for 4 agents
        metadata[8:12] = self.agent_types_np[:4] # TEMP -> harcoded for 4 agents
        metadata[12:16] = self.has_flag[:4]
        #metadata[16:18] = [v for v in self.metrics['team_points'].values()]
        #metadata[18:22] = [self.agent_hp[kv[1]]/self.AGENT_TYPE_HP[self.AGENT_TYPES[i]] for i, kv in enumerate(self.AGENT_TYPES.items())]

        return metadata

    def play(self, 
             player=0, 
             agents=None, 
             use_ego_state=False, 
             scale_tiles=False, 
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

        if use_ego_state:
            env_dims = self.EGO_ENV_DIMS
        else:
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
                    
                    grid_state_ = self.standardise_state(agent_idx, use_ego_state=use_ego_state, scale_tiles=scale_tiles).reshape(*env_dims) + ut.add_noise(env_dims)
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