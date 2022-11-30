import numpy as np

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

# Tile values for each agent
AGENT_TILE_MAP = {
    0: 2,
    1: 3,
    2: 4,
    3: 5
}

# Position of flags for each team
FLAG_POSITIONS = {
    0: (1, 1),
    1: (8, 8)
}

# Tile values for flags
FLAG_TILE_MAP = {
    0: 6,
    1: 7
}

# Capture positions for each team
CAPTURE_POSITIONS = {
    0: (1, 1),
    1: (8, 8)
}

# Colour map for rendering environment 
COLOUR_MAP = {0: np.array([224, 224, 224]), # light grey
            1: np.array([0, 0, 0]), # black
            2: np.array([0, 128, 255]), # blue 
            3: np.array([0, 128, 255]), # blue
            4: np.array([255, 51, 0]), # red
            5: np.array([255, 51, 0]), # red
            6: np.array([0, 0, 153]), # dark blue
            7: np.array([153, 0, 0]) # dark red
}  

# TODO: add map