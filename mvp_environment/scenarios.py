import numpy as np

class CtfScenarios:

    donut = {
        'SCENARIO_NAME': 'Donut',
        'GRID_SIZE': 11,        
        'FLIP_AXIS': None,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 1),
            1: (9, 9)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 1),
            1: (9, 9)
        },
        'SPAWN_POSITIONS' : {
            0: (1, 9),
            1: (9, 1)
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (2, 8),
            1: (8, 2),
            2: (2, 9),
            3: (8, 1),
            4: (2, 10),
            5: (8, 0),
        },
        'BLOCK_TILE_SLICES' : [
            (slice(4, 7), slice(4, 7)),
            (3, 5),
            (5, 3),
            (7, 5),
            (5, 7)
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
        ]
    }

    arrow = {
        'SCENARIO_NAME': 'Arrow',
        'GRID_SIZE': 11,        
        'FLIP_AXIS': 2,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 1),
            1: (9, 9)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 1),
            1: (9, 9)
        },
        'SPAWN_POSITIONS' : {
            0: (6, 1),
            1: (9, 4)
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (5, 2),
            1: (8, 5),
            2: (5, 1),
            3: (9, 5),
            4: (5, 0),
            5: (10, 5),
        },
        'BLOCK_TILE_SLICES' : [
            (10, 0),
            (9, 1),
            (8, 2),
            (7, 3),
            (6, 4),
            (5, 5)
            # (4, 5),
            # (5, 6)
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
        ]
    }

    the_fence = {
        'SCENARIO_NAME': 'The Fence',
        'GRID_SIZE': 11,        
        'FLIP_AXIS': 0,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 1),
            1: (9, 1)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 1),
            1: (9, 1)
        },
        'SPAWN_POSITIONS' : {
            0: (1, 9),
            1: (9, 9)
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (2, 8),
            1: (8, 8),
            2: (2, 9),
            3: (8, 9),
            4: (2, 10),
            5: (8, 10),
        },
        'BLOCK_TILE_SLICES' : [
            (5, slice(0, 4)),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
        ]
    }

    jailbreak = {
        'SCENARIO_NAME': 'Jailbreak',
        'GRID_SIZE': 11,        
        'FLIP_AXIS': None,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 1),
            1: (9, 9)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 1),
            1: (9, 9)
        },
        'SPAWN_POSITIONS' : {
            0: (1, 9),
            1: (9, 1)
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (0, 5),
            1: (10, 5),
            2: (2, 8),
            3: (8, 2),
            4: (2, 9),
            5: (8, 1),
        },
        'BLOCK_TILE_SLICES' : [
            (5, slice(0, 2)),
            (5, slice(9, 11)),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            (0, 4),
            (0, 6),
            (1, slice(4, 7)),
            (10, 4),
            (10, 6),
            (9, slice(4, 7)),
        ]
    }

    one_way_out = {
        'SCENARIO_NAME': 'One way out',
        'GRID_SIZE': 11,        
        'FLIP_AXIS': 0,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 1),
            1: (9, 1)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 1),
            1: (9, 1)
        },
        'SPAWN_POSITIONS' : {
            0: (1, 9),
            1: (9, 9)
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (2, 8),
            1: (8, 8),
            2: (2, 9),
            3: (8, 9),
            4: (2, 10),
            5: (8, 10),
        },
        'BLOCK_TILE_SLICES' : [
            (slice(0, 3), 6),
            (slice(8, 11), 6),
            (5, slice(7, 11)),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            (5, slice(0, 3)),
            (4, slice(7, 11)),
            (6, slice(7, 111)),
        ]
    }

    keyhole = {
        'SCENARIO_NAME': 'Keyhole',
        'GRID_SIZE': 11,        
        'FLIP_AXIS': 0,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 1),
            1: (9, 1)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 1),
            1: (9, 1)
        },
        'SPAWN_POSITIONS' : {
            0: (1, 9),
            1: (9, 9)
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (2, 8),
            1: (8, 8),
            2: (2, 9),
            3: (8, 9),
            4: (2, 10),
            5: (8, 10),
        },
        'BLOCK_TILE_SLICES' : [
            (5, slice(0, 3)),
            (slice(4, 7), 0),
            (slice(4, 7), 2),
            (slice(4, 7), slice(7, 11)),
            (slice(0, 3), 6),
            (slice(8, 11), 6),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            (slice(4, 7), 6),
        ]
    }    

       
