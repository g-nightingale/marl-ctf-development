import numpy as np

#----------------------------------------------------------------------------------
# General Config
#----------------------------------------------------------------------------------

# Grid legnth
GRID_LEN = 10

# Rewards
REWARD_CAPTURE = 100
REWARD_STEP = -1
WINNING_POINTS = 1

# Toggle pickups
ENABLE_PICKUPS = True

# Agents and actions
N_AGENTS = 4
N_ACTIONS = 4

# Position of flags for each team
FLAG_POSITIONS = {
    0: (1, 1),
    1: (8, 8)
}

# Capture positions for each team
CAPTURE_POSITIONS = {
    0: (1, 1),
    1: (8, 8)
}

# Block counts for constructor agents
BLOCK_PICKUP_VALUE = 1
HEALTH_PICKUP_VALUE = 1
MAX_HEALTH_PICKUP_TILES = 1

MAX_PERCENT_NON_OPEN_TILES = 0.3

#----------------------------------------------------------------------------------
# Agent Config
#----------------------------------------------------------------------------------

# Starting positions for each agent
AGENT_STARTING_POSITIONS = {
    0: (0, 2),
    1: (2, 0),
    2: (7, 9),
    3: (9, 7),
}

# Teams that agent's belong to: 0 = team 1, 1 = team 2
AGENT_TEAMS = {
    0: 0,
    1: 0,
    2: 1,
    3: 1
}

# Agent types: 0 = flag carrier, 1 = tagger, 2 = constructor
AGENT_TYPES = {
    0: 0,
    1: 1,
    2: 0,
    3: 1
}

# Agent hitpoints
AGENT_TYPE_HP = {
    0: 3,
    1: 1,
    2: 2
}

# Probability of a successful tag for a tagging agent
TAG_PROBABILITY = 0.8

# Agent flag capture types - types that can capture the flag
AGENT_FLAG_CAPTURE_TYPES = [0, 1, 2]

MAX_AGENT_HEALTH = 5
MAX_AGENT_BLOCKS = 5

#----------------------------------------------------------------------------------
# Tile definitions
#----------------------------------------------------------------------------------
# Open and block tiles
OPEN_TILE = 0
BLOCK_TILE = 1
DESTRUCTIBLE_TILE1 = 2
DESTRUCTIBLE_TILE2 = 3
DESTRUCTIBLE_TILE3 = 4

# Tile values for each agent
AGENT_TILE_MAP = {
    0: 5,
    1: 6,
    2: 7,
    3: 8
}

# Tile values for flags
FLAG_TILE_MAP = {
    0: 9,
    1: 10
}

# Block pickup tile
HEALTH_PICKUP_TILE = 11

# Standardised tiles
STD_EGO_AGENT_TILE = 10
STD_AGENT_TYPE_TILES = {
    0: 20,
    1: 30,
    2: 40,
}
STD_BLOCK_TILE = 50
STD_DESTRUCTIBLE_TILE1 = 60
STD_DESTRUCTIBLE_TILE2 = 70
STD_DESTRUCTIBLE_TILE3 = 80
STD_OWN_FLAG_TILE = 90
STD_OPP_FLAG_TILE = 100
STD_HEALTH_PICKUP_TILE = 110

# Colour map for rendering environment. See: https://www.rapidtables.com/web/color/RGB_Color.html
COLOUR_MAP = {
            0: np.array([224, 224, 224]), # light grey
            1: np.array([0, 0, 0]), # black
            2: np.array([32, 32, 32]), # black
            3: np.array([64, 64, 64]), # black
            4: np.array([128, 128, 128]), # black
            5: np.array([102, 178, 255]), # blue 
            6: np.array([0, 128, 255]), # blue
            7: np.array([255, 102, 102]), # red
            8: np.array([255, 51, 0]), # red
            9: np.array([0, 0, 153]), # dark blue
            10: np.array([153, 0, 0]), # dark red
            11: np.array([153, 51, 255]) # purple
}  
