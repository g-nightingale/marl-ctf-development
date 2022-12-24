import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import env_config as cfg
import utils as ut
import torch

class GridworldCtf:
    """
    A GridWorld capture the flag environment.
    
    """

    def __init__(self, game_mode='static') -> None:
        self.GAME_MODE = game_mode # static or random

        # Load config from file
        self.MAPS = cfg.MAPS
        self.DEFAULT_MAP = cfg.DEFAULT_MAP
        self.GRID_LEN = cfg.GRID_LEN
        self.REWARD_STEP = cfg.REWARD_STEP
        self.REWARD_CAPTURE = cfg.REWARD_CAPTURE
        self.WINNING_POINTS = cfg.WINNING_POINTS
        self.N_AGENTS = cfg.N_AGENTS
        self.N_ACTIONS = cfg.N_ACTIONS
        self.OPEN_TILE = cfg.OPEN_TILE
        self.BLOCK_TILE = cfg.BLOCK_TILE
        self.PLACEHOLDER_TILE = cfg.PLACEHOLDER_TILE
        self.AGENT_STARTING_POSITIONS = cfg.AGENT_STARTING_POSITIONS
        self.FLAG_POSITIONS = cfg.FLAG_POSITIONS
        self.FLAG_TILE_MAP = cfg.FLAG_TILE_MAP
        self.AGENT_TILE_MAP = cfg.AGENT_TILE_MAP
        self.AGENT_TEAMS = cfg.AGENT_TEAMS
        self.AGENT_TYPES = cfg.AGENT_TYPES
        self.TAG_PROBABILITY = cfg.TAG_PROBABILITY
        self.CAPTURE_POSITIONS = cfg.CAPTURE_POSITIONS
        self.COLOUR_MAP = cfg.COLOUR_MAP
        self._arr = np.arange(self.N_AGENTS)

        self.reset()

    def reset(self):
        """ 
        Reset the environment. 
        """

        self.agent_positions = self.AGENT_STARTING_POSITIONS.copy()
        self.OPPONENTS = {
            0: [k for k, v in self.AGENT_TEAMS.items() if v==1],
            1: [k for k, v in self.AGENT_TEAMS.items() if v==0],
        }
        
        if self.GAME_MODE=='static':
            self.grid = self.MAPS[self.DEFAULT_MAP].copy()
        elif self.GAME_MODE =='random':
            self.grid = self.MAPS[np.random.randint(4)].copy()
        else:
            raise ValueError("Game mode must be set to either 'static' or 'random'")

        self.init_objects()
        self.has_flag = np.zeros(self.N_AGENTS, dtype=np.int8)
        self.agent_types_np = np.array([v for v in self.AGENT_TYPES.values()])
        self.done = False
        self.team_points = {0:0, 1:0}
        self.tag_count = {0:0, 1:0}

    def init_objects(self) -> np.array:
        """"
        Initialise the grid.
        """

        # Add agents
        self.grid[self.AGENT_STARTING_POSITIONS[0]] = self.AGENT_TILE_MAP[0]
        self.grid[self.AGENT_STARTING_POSITIONS[1]]  = self.AGENT_TILE_MAP[1]

        self.grid[self.AGENT_STARTING_POSITIONS[2]] = self.AGENT_TILE_MAP[2]
        self.grid[self.AGENT_STARTING_POSITIONS[3]]  = self.AGENT_TILE_MAP[3]

        # Add flags
        self.grid[self.FLAG_POSITIONS[0]] = self.FLAG_TILE_MAP[0]
        self.grid[self.FLAG_POSITIONS[1]] = self.FLAG_TILE_MAP[1]


    def act(self, agent_idx, action) -> None:
        """
        Take agent actions and update grid.
        """

        # Get the current position of the agent
        curr_pos = self.agent_positions[agent_idx]

        #--------------------------------------------------------------------------------------------
        # Movement actions for all agents
        #--------------------------------------------------------------------------------------------
        # Move up
        if action == 0:
            if curr_pos[0] > 0 \
              and self.grid[curr_pos[0] - 1, curr_pos[1]] == self.OPEN_TILE:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0] - 1, curr_pos[1])
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Move down
        elif action == 1:
            if curr_pos[0] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0] + 1, curr_pos[1]] == self.OPEN_TILE:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0] + 1, curr_pos[1])
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Move right
        elif action == 2:
            if curr_pos[1] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0], curr_pos[1] + 1] == self.OPEN_TILE:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0], curr_pos[1] + 1)
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Move left
        elif action == 3:
            if curr_pos[1] > 0 \
              and self.grid[curr_pos[0], curr_pos[1] - 1] == self.OPEN_TILE:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0], curr_pos[1] - 1)
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        #--------------------------------------------------------------------------------------------
        # Actions specific to builder agent
        #--------------------------------------------------------------------------------------------
        # Add block up top
        elif action == 4:
            if curr_pos[0] > 0 \
              and self.grid[curr_pos[0] - 1, curr_pos[1]] == self.OPEN_TILE:
                self.grid[curr_pos[0] - 1, curr_pos[1]] = self.BLOCK_TILE
        # Add block on bottom
        elif action == 5:
            if curr_pos[0] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0] + 1, curr_pos[1]] == self.OPEN_TILE:
                self.grid[curr_pos[0] + 1, curr_pos[1]] == self.BLOCK_TILE
        # Add block to right
        elif action == 6:
            if curr_pos[1] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0], curr_pos[1] + 1] == self.OPEN_TILE:
                self.grid[curr_pos[0], curr_pos[1] + 1] == self.BLOCK_TILE
        # Add block to left
        elif action == 7:
            if curr_pos[1] > 0 \
              and self.grid[curr_pos[0], curr_pos[1] - 1] == self.OPEN_TILE:
                self.grid[curr_pos[0], curr_pos[1] - 1] == self.BLOCK_TILE
        #--------------------------------------------------------------------------------------------
        # Actions specific to deconstructor agent
        #--------------------------------------------------------------------------------------------
        # Remove block up top
        elif action == 8:
            if curr_pos[0] > 0 \
              and self.grid[curr_pos[0] - 1, curr_pos[1]] == self.BLOCK_TILE:
                self.grid[curr_pos[0] - 1, curr_pos[1]] = self.OPEN_TILE
        # Remove block on bottom
        elif action == 9:
            if curr_pos[0] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0] + 1, curr_pos[1]] == self.BLOCK_TILE:
                self.grid[curr_pos[0] + 1, curr_pos[1]] == self.OPEN_TILE
        # Remove block to right
        elif action == 10:
            if curr_pos[1] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0], curr_pos[1] + 1] == self.BLOCK_TILE:
                self.grid[curr_pos[0], curr_pos[1] + 1] == self.OPEN_TILE
        # Remove block to left
        elif action == 11:
            if curr_pos[1] > 0 \
              and self.grid[curr_pos[0], curr_pos[1] - 1] == self.BLOCK_TILE:
                self.grid[curr_pos[0], curr_pos[1] - 1] == self.OPEN_TILE

        # Do nothing -> removed for now
        # elif action == 12:
        #     pass

        # Update agent position
        self.agent_positions[agent_idx] = curr_pos
       

    def dice_roll(self) -> np.array:
        """
        Determines order that agents make moves.
        """

        np.random.shuffle(self._arr)

        return self._arr

    def check_object_distance(self, agent_idx, object_tile) -> bool:
        """
        Check if agent is within one cell of a given object.
        """
        
        x, y = self.agent_positions[agent_idx]
        capture = self.grid[max(x-1, 0):x+2, max(y-1, 0):y+2] == object_tile

        return capture.any()

    def check_object_distance2(self, agent_idx, object_xy) -> bool:
        """
        Check if agent is within one cell of a given object.
        """
        
        x, y = self.agent_positions[agent_idx]
        capture = np.abs(np.array([x, y]) - np.array([object_xy[0], object_xy[1]]))
        return max(capture) <= 1

    def respawn(self, agent_idx):
        """
        Respawn agent near flag if tagged.
        """

        # Get agent team flag position - we will respawn around there
        x, y = self.FLAG_POSITIONS[self.AGENT_TEAMS[agent_idx]]
        possible_respawn_offset = np.where(self.grid[max(x-1, 0):x+2, max(y-1, 0):y+2]==self.OPEN_TILE)

        # Choose an open respawn location at random
        rnd = np.random.randint(possible_respawn_offset[0].shape[0])
        
        # Get respawn position: WARNING - if the flag is at (0, 0) this will break 
        new_pos = (x + possible_respawn_offset[0][rnd] - 1, y + possible_respawn_offset[1][rnd] - 1)

        # Update tiles
        self.grid[self.agent_positions[agent_idx]] = self.OPEN_TILE
        self.grid[new_pos] = self.AGENT_TILE_MAP[agent_idx]
        
        # Update agent position
        self.agent_positions[agent_idx] = new_pos

        # Reset flag if agent has it
        if self.has_flag[agent_idx]==1:
            self.has_flag[agent_idx] = 0
            self.grid[self.FLAG_POSITIONS[1-self.AGENT_TEAMS[agent_idx]]] = self.FLAG_TILE_MAP[1-self.AGENT_TEAMS[agent_idx]]


    def step(self, actions) -> tuple:
        """
        Take a step in the environment.

        Takes in a vector or actions, where each position in the vector represents an agent,
        and the value represents the action to take.
        """

        rewards = [0, 0, 0, 0]
        for agent_idx in self.dice_roll():
            # Initialise reward counter
            reward = 0

            # Get the agent team
            agent_team = self.AGENT_TEAMS[agent_idx]

            # Get the agent action
            action = actions[agent_idx]

            # Move the agent
            self.act(agent_idx, action)

            # If agent passes over flag square, set to zero and mark agent as having the flag
            if self.AGENT_TYPES[agent_idx]==0 and self.check_object_distance(agent_idx, self.FLAG_TILE_MAP[1-agent_team]):
                self.has_flag[agent_idx] = 1
                self.grid[self.FLAG_POSITIONS[1-agent_team]] = self.PLACEHOLDER_TILE
            # Calculate rewards - game is done when the agent has the flag and reaches the capture position
            elif self.AGENT_TYPES[agent_idx]==0 and self.has_flag[agent_idx] == 1 and self.check_object_distance2(agent_idx, self.CAPTURE_POSITIONS[agent_team]):
                reward += self.REWARD_CAPTURE
                self.has_flag[agent_idx] = 0
                self.grid[self.FLAG_POSITIONS[1-agent_team]] = self.FLAG_TILE_MAP[1-agent_team]
                self.team_points[agent_team] += 1
                if self.team_points[agent_team] == self.WINNING_POINTS:
                    self.done = True
            
            # Check for tag
            if self.AGENT_TYPES[agent_idx]==1: 
                for opp_agent_idx in self.OPPONENTS[agent_team]:
                    if self.check_object_distance2(agent_idx, self.agent_positions[opp_agent_idx]):
                        if np.random.rand() < self.TAG_PROBABILITY:
                            self.respawn(opp_agent_idx)
                            self.tag_count[agent_team] += 1

            # Step reward is applied in all situations
            reward += self.REWARD_STEP

            # Update agent specific reward
            rewards[agent_idx] = reward

        return self.grid, rewards, self.done

    def display_grid(self) -> None:
        """
        Display the current grid.
        """

        print(self.grid, '\n')

    def render(self, sleep_time: float=0.2) :
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
        env_plot = np.copy(self.grid)

        # make a 3d numpy array that has a color channel dimension   
        data_3d = np.ndarray(shape=(env_plot.shape[0], env_plot.shape[1], 3), dtype=int)
        for i in range(0, env_plot.shape[0]):
            for j in range(0, env_plot.shape[1]):
                data_3d[i][j] = self.COLOUR_MAP[env_plot[i][j]]

        # Plot the gridworld.
        ax.imshow(data_3d)

        # Set up axes.
        ax.grid(which='major', axis='both', linestyle='-', color='0.4', linewidth=2, zorder=1)
        ax.set_xticks(np.arange(-0.5, self.grid.shape[1] , 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1))
        ax.set_yticklabels([])

        plt.show()

        # Sleep if desired.
        if (sleep_time > 0) :
            time.sleep(sleep_time)

    def play(self, player=0, agents=None, env_dims=(1, 1, 10, 10), device='cpu'):
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
                    }

        raw_action = None

        self.render()

        # start main loop
        while True:  
            print(f"Move {move_counter}")
            
            if move_counter > 1:
                print(f"reward: {rewards[0]}, done: {done}")

            while raw_action not in ['w', 's', 'a', 'd', 'x']:
                raw_action = input("Enter an action")

            if raw_action == 'x':
                print(f"Game exited")
                break

            # initialise random actions
            actions = np.random.randint(4, size=4)

            # if agents are supplied, get agent actions
            if agents is not None:
                grid_state_ = self.grid.reshape(*env_dims) + ut.add_noise(env_dims)
                grid_state = torch.from_numpy(grid_state_).float().to(device)

                for agent_idx in np.arange(4):
                    metadata_state = ut.get_env_metadata(agent_idx, self.has_flag, self.agent_types_np)
                    actions[agent_idx] = agents[agent_idx].choose_action(grid_state, metadata_state)

            # insert player action
            actions[player] = int(ACTIONS_MAP[raw_action])

            # step the environment
            _, rewards, done = self.step(actions)

            total_score += rewards[0]
            move_counter += 1

            raw_action = None
            self.render()

            if done:
                print(f"You win!, total score {total_score}")
                break