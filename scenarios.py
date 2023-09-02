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

    skittles = {
        'SCENARIO_NAME': 'Skittles',
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
            0: (1, 8),
            1: (9, 2),
            2: (2, 9),
            3: (8, 1),
        },
        'BLOCK_TILE_SLICES' : [
            # Blue flag zone
            (slice(0, 2), 3),
            (3, slice(0, 2)),
            # Blue spawn zone
            (slice(0, 2), 7),
            (3, slice(9, 11)),            
            # Red flag zone
            (slice(9, 11), 7),
            (7, slice(9, 11)),
            # Red spawn zone
            (slice(9, 11), 3),
            (7, slice(0, 2)),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            (1, 5),
            (3, 3),
            (3, 5),
            (3, 7),
            (5, 1),
            (5, 3),
            (5, 5),
            (5, 7),
            (5, 9),
            (7, 3),
            (7, 5),
            (7, 7),
            (9, 5)
        ]
    }

    the_wall = {
        'SCENARIO_NAME': 'The Wall',
        'GRID_SIZE': 13,        
        'FLIP_AXIS': None,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (1, 6),
            1: (11, 6)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (1, 6),
            1: (11, 6)
        },
        'SPAWN_POSITIONS' : {
            0: (5, 11),
            1: (7, 1),
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (4, 11),
            1: (8, 1),
            2: (5, 10),
            3: (7, 2),
            4: (5, 12),
            5: (7, 0),
        },
        'BLOCK_TILE_SLICES' : [
            # Red spawn zone borders
            (5, slice(2, 4)),
            (6, 3),
            # Blue spawn zone borders
            (7, slice(9, 11)),
            (6, 9),
            # LHS walls
            (slice(0, 3), slice(0, 2)),
            (slice(10, 13), slice(0, 2)),
            (4, 1),
            # RHS walls
            (slice(0, 3), slice(11, 13)),
            (slice(10, 13), slice(11, 13)),
            (8, 11),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            # Shortcuts
            (3, slice(0, 2)),
            (9, slice(11, 13)),
            # The Wall
            (slice(5, 8), slice(4, 6)),
            (slice(5, 8), slice(7, 9)),
            (6, 6)
        ]
    }

    gridlocked = {
        'SCENARIO_NAME': 'Gridlocked',
        'GRID_SIZE': 13,        
        'FLIP_AXIS': None,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (2, 4),
            1: (10, 8)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (2, 4),
            1: (10, 8)
        },
        'SPAWN_POSITIONS' : {
            0: (1, 11),
            1: (11, 1),
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (2, 11),
            1: (10, 1),
            2: (1, 10),
            3: (11, 2),
            4: (1, 12),
            5: (11, 0),
        },
        'BLOCK_TILE_SLICES' : [
            # Red spawn zone borders
            (9, slice(0, 3)),
            (slice(11, 13), 3),
            # Blue spawn zone borders
            (3, slice(10, 13)),
            (slice(0, 2), 9),
            # Blue flag zone
            # (2, 3),
            # (2, 5),
            # Red flag zone
            # (10, 7),
            # (10, 9),
        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            # The grid
            (4, 2),
            (4, 4),
            (4, 6),
            (4, 8),
            (4, 10),
            (6, 2),
            (6, 4),
            (6, 6),
            (6, 8),
            (6, 10),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (8, 10),
        ]
    }

    arena = {
        'SCENARIO_NAME': 'Arena',
        'GRID_SIZE': 15,        
        'FLIP_AXIS': None,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (2, 7),
            1: (12, 7)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (2, 7),
            1: (12, 7)
        },
        'SPAWN_POSITIONS' : {
            0: (5, 13),
            1: (9, 1),
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (5, 12),
            1: (9, 2),
            2: (6, 13),
            3: (8, 1),
            4: (4, 13),
            5: (10, 1),
            6: (0, 1),
            7: (14, 13)
        },
        'BLOCK_TILE_SLICES' : [
            # Centrepiece
            (7, 7),
            # Blue spawn zone borders
            (3, slice(13, 15)),
            (slice(5, 8), 11),
            (7, slice(12, 15)),
            # Red spawn zone borders
            (7, slice(0, 4)),
            (slice(8, 10), 3),
            (11, slice(0, 2)),

        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            # Blue flag borders
            (2, slice(5, 7)),
            (2, slice(8, 10)),
            # Red flag borders
            (12, slice(5, 7)),
            (12, slice(8, 10)),
            # Team 1 trapped agent
            (14, 12),
            (14, 14),
            (13, slice(12, 15)),
            # Team 2 trapped agent
            (0, 0),
            (0, 2),
            (1, slice(0, 3)),
            # Block caches
            (6, slice(0, 2)),
            (8, slice(13, 15)),
            # Centre blocks
            (4, 5),
            (4, 9),
            (5, 7),
            (7, 5),
            (7, 9),
            (9, 7),
            (10, 5),
            (10, 9),
        ]
    }

    arena_ii = {
        'SCENARIO_NAME': 'Arena II',
        'GRID_SIZE': 15,        
        'FLIP_AXIS': None,
        # Team level config
        'FLAG_POSITIONS' : {
            0: (2, 7),
            1: (12, 7)
        },
        # Capture positions for each team
        'CAPTURE_POSITIONS' : {
            0: (2, 7),
            1: (12, 7)
        },
        'SPAWN_POSITIONS' : {
            0: (5, 13),
            1: (9, 1),
        },
        # Starting positions for each agent
        'AGENT_STARTING_POSITIONS' : {
            0: (5, 12),
            1: (9, 2),
            2: (6, 13),
            3: (8, 1),
            4: (4, 13),
            5: (10, 1),
            6: (5, 14),
            7: (9, 0)
        },
        'BLOCK_TILE_SLICES' : [
            # Centrepiece
            (7, 7),
            # Blue spawn zone borders
            (3, slice(13, 15)),
            (slice(5, 8), 11),
            (7, slice(12, 15)),
            # Red spawn zone borders
            (7, slice(0, 4)),
            (slice(8, 10), 3),
            (11, slice(0, 2)),

        ],
        'DESTRUCTIBLE_TILE_SLICES' : [
            # Blue flag borders
            (2, slice(5, 7)),
            (2, slice(8, 10)),
            # Red flag borders
            (12, slice(5, 7)),
            (12, slice(8, 10)),
            # Team 1 trapped agent
            (12, slice(12, 15)),
            # Team 2 trapped agent
            (2, slice(0, 3)),
            # Block caches
            (6, 0),
            (6, 2),
            (8, 12),
            (8, 14),
            # Centre blocks
            (4, 5),
            (4, 9),
            (5, 7),
            (7, 5),
            (7, 9),
            (9, 7),
            (10, 5),
            (10, 9),
        ]
    }

    arena_iii = {
            'SCENARIO_NAME': 'Arena III',
            'GRID_SIZE': 15,        
            'FLIP_AXIS': None,
            # Team level config
            'FLAG_POSITIONS' : {
                0: (2, 7),
                1: (12, 7)
            },
            # Capture positions for each team
            'CAPTURE_POSITIONS' : {
                0: (2, 7),
                1: (12, 7)
            },
            'SPAWN_POSITIONS' : {
                0: (5, 13),
                1: (9, 1),
            },
            # Starting positions for each agent
            'AGENT_STARTING_POSITIONS' : {
                0: (5, 12),
                1: (9, 2),
                2: (6, 13),
                3: (8, 1),
                4: (4, 13),
                5: (10, 1),
                6: (5, 14),
                7: (9, 0)
            },
            'BLOCK_TILE_SLICES' : [
                # Centrepiece
                (7, 7),
                # Blue spawn zone borders
                (2, slice(13, 15)),
                (5, 11),
                (6, 11),
                (7, 12),
                # Red spawn zone borders
                (7, 2),
                (8, 3),
                (9, 3),
                (12, slice(0, 2)),
                # Flag borders
                (2, 6),
                (2, 8),
                (12, 6),
                (12, 8),
                # Walls near flags
                (12, slice(12, 15)),
                (2, slice(0, 3)),
            ],
            'DESTRUCTIBLE_TILE_SLICES' : [
                # Blue flag borders
                (2, 5),
                (2, 9),
                # Red flag borders
                (12, 5),
                (12, 9),
                # Block caches
                (6, 0),
                (6, 1),
                (8, 13),
                (8, 14),
                # Centre blocks
                (4, 5),
                (4, 9),
                (5, 7),
                (7, 4),
                (7, 10),
                (9, 7),
                (10, 5),
                (10, 9),
            ]
        }
