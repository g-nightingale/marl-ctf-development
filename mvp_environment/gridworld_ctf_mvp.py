import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import env_config as cfg

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
        self.N_AGENTS = cfg.N_AGENTS
        self.N_ACTIONS = cfg.N_ACTIONS
        self.OPEN_TILE = cfg.OPEN_TILE
        self.BLOCK_TILE = cfg.BLOCK_TILE
        self.AGENT_STARTING_POSITIONS = cfg.AGENT_STARTING_POSITIONS
        self.FLAG_POSITIONS = cfg.FLAG_POSITIONS
        self.FLAG_TILE_MAP = cfg.FLAG_TILE_MAP
        self.AGENT_TILE_MAP = cfg.AGENT_TILE_MAP
        self.AGENT_TEAMS = cfg.AGENT_TEAMS
        self.CAPTURE_POSITIONS = cfg.CAPTURE_POSITIONS
        self.COLOUR_MAP = cfg.COLOUR_MAP
        self._arr = np.arange(self.N_AGENTS)

        self.reset()

    def reset(self):
        """ 
        Reset the environment. 
        """

        self.agent_positions = self.AGENT_STARTING_POSITIONS
        
        if self.GAME_MODE=='static':
            self.grid = self.MAPS[self.DEFAULT_MAP]
        elif self.GAME_MODE =='random':
            self.grid = self.MAPS[np.random.randint(4)]
        else:
            raise ValueError("Game mode must be set to either 'static' or 'random'")

        self.init_objects()
        self.has_flag = np.zeros(self.N_AGENTS, dtype=np.int8)
        self.done = False

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
        # Do nothing
        elif action == 4:
            pass

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


    def step(self, actions) -> tuple:
        """
        Take a step in the environment.

        Takes in a vector or actions, where each position in the vector represents an agent,
        and the value represents the action to take.
        """

        rewards = [0, 0, 0, 0]
        for agent_idx in self.dice_roll():
            agent_team = self.AGENT_TEAMS[agent_idx]

            # Get the agent action
            action = actions[agent_idx]

            # Move the agent
            self.act(agent_idx, action)

            # If agent passes over flag square, set to zero and mark agent as having the flag
            if self.check_object_distance(agent_idx, self.FLAG_TILE_MAP[1-agent_team]):
                self.has_flag[agent_idx] = 1
                self.grid[self.FLAG_POSITIONS[1-agent_team]] = 0
                reward = self.REWARD_STEP
            # Calculate rewards - game is done when the agent has the flag and reaches the capture position
            elif self.has_flag[agent_idx] == 1 and self.check_object_distance2(agent_idx, self.CAPTURE_POSITIONS[agent_team]):
                reward = self.REWARD_CAPTURE
                self.done = True
            else:
                reward = self.REWARD_STEP

            # Update agent specific reward
            rewards[agent_idx] = reward

        return self.grid, rewards, self.done

    def display_grid(self) -> None:
        """
        Display the current grid.
        """

        print(self.grid, '\n')

    def render(self, sleep_time: float=0.1) :
        """
        Renders a pretty matplotlib plot representing the current state of the environment.
        Calling this method on subsequent timesteps will update the plot.
        This is VERY VERY SLOW and wil slow down training a lot. Only use for debugging/testing.

        Arguments:
            sleep_time {float} -- How many seconds (or partial seconds) you want to wait on this rendered frame.

        """
        # Turn interactive mode on.
        plt.ion()
        fig = plt.figure(num="env_render")
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment plot and mark the car's position.
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

        # Draw everything.
        #fig.canvas.draw()
        #fig.canvas.flush_events()

        #plt.show()

        # Sleep if desired.
        if (sleep_time > 0) :
            time.sleep(sleep_time)

    def play(self):
        """
        Play the environment manually.
        """

        return None

        # self.reset()
        # move_counter = 0
        # total_score = 0

        # ACTIONS_MAP = {
        #                 'w':0,
        #                 's':1,
        #                 'd':2,
        #                 'a':3,
        #             }

        # raw_action = None

        # while True:  
        #     print(f"Move {move_counter}")
        #     self.render()

        #     if move_counter > 1:
        #         print(f"reward: {reward}, done: {done}")

        #     while raw_action not in ['w', 's', 'a', 'd', 'x']:
        #         raw_action = input("Enter an action")

        #     if raw_action == 'x':
        #         print(f"Game exited")
        #         break

        #     action = ACTIONS_MAP[raw_action]

        #     _, reward, done = self.step(int(action))

        #     total_score += reward
        #     move_counter += 1

        #     raw_action = None

        #     if done:
        #         print(f"You win!, total score {total_score}")
        #         break