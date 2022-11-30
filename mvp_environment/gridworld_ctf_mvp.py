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

    def __init__(self, agent_list, game_mode='static') -> None:
        self.GAME_MODE = game_mode # static or random
        self.agent_list = agent_list
        self.BLOCK_NUM = 1
        self.GRID_LEN = 10
        self.REWARD_GOAL = 100
        self.REWARD_STEP = -1
        self.REWARD_CAPTURE = -1
        self.N_ACTIONS = 5
        self.AGENT_STARTING_POSITIONS = cfg.AGENT_STARTING_POSITIONS
        self.FLAG_POSITIONS = cfg.FLAG_POSITIONS
        self.FLAG_TILE_MAP = cfg.FLAG_TILE_MAP
        self.AGENT_TILE_MAP = cfg.AGENT_TILE_MAP
        self.AGENT_TEAMS = cfg.AGENT_TEAMS
        self.COLOUR_MAP = cfg.COLOUR_MAP

        self.reset()

    def reset(self):
        """ 
        Reset the environment. 
        """
        self.grid = np.flip(np.loadtxt("./map.txt", dtype=np.int8), axis = 0)

        if self.GAME_MODE=='static':
            self.agent_positions = self.AGENT_STARTING_POSITIONS.copy()


        elif self.GAME_MODE =='random':
            # self.agent_position = (np.random.randint(7, 10), np.random.randint(10))
            # self.agent_start_pos = self.agent_position
            # self.flag_pos = (np.random.randint(0, 3), np.random.randint(10))
            # capture_pos = None
            # while capture_pos is None or capture_pos == self.agent_position:
            #     capture_pos = (np.random.randint(7, 10), np.random.randint(10))
            # self.capture_pos = capture_pos
            raise ValueError("Not implemented")
        else:
            raise ValueError("Game mode must be set to either 'static' or 'random'")

        self.init_objects()
        self.has_flag = False
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
        Take agent actions.
        """

        # Get the current position of the agent
        curr_pos = self.agent_positions[agent_idx]
        team = self.AGENT_TEAMS[agent_idx]

        # Move up
        if action == 0:
            if curr_pos[0] > 0 \
              and self.grid[curr_pos[0] - 1, curr_pos[1]] != self.BLOCK_NUM \
              and self.grid[curr_pos[0] - 1, curr_pos[1]] != self.FLAG_TILE_MAP[team]:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0] - 1, curr_pos[1])
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Move down
        elif action == 1:
            if curr_pos[0] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0] + 1, curr_pos[1]] != self.BLOCK_NUM \
              and self.grid[curr_pos[0] + 1, curr_pos[1]] != self.FLAG_TILE_MAP[team]:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0] + 1, curr_pos[1])
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Move right
        elif action == 2:
            if curr_pos[1] < (self.GRID_LEN-1) \
              and self.grid[curr_pos[0], curr_pos[1] + 1] != self.BLOCK_NUM \
              and self.grid[curr_pos[0], curr_pos[1] + 1] != self.FLAG_TILE_MAP[team]:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0], curr_pos[1] + 1)
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Move left
        elif action == 3:
            if curr_pos[1] > 0 \
              and self.grid[curr_pos[0], curr_pos[1] - 1] != self.BLOCK_NUM \
              and self.grid[curr_pos[0], curr_pos[1] - 1] != self.FLAG_TILE_MAP[team]:
                self.grid[curr_pos] = 0
                curr_pos = (curr_pos[0], curr_pos[1] - 1)
                self.grid[curr_pos] = self.AGENT_TILE_MAP[agent_idx]
        # Do nothing
        elif action == 4:
            pass

        # Update agent position
        self.agent_positions[agent_idx] = curr_pos
       

    def dice_roll(self):
        """
        Determines order that agents make moves.
        """

        arr = np.arange(len(self.agent_list))
        np.random.shuffle(arr)

        return arr

    def step(self, actions=None) -> tuple:
        """
        Take a step in the environment.
        """

        # Randomly select order of agent actions
        for agent_idx in self.dice_roll():
            print(f"Agent {agent_idx}'s move")

            # Get the agent action
            action = np.random.randint(5)

            # Move the agent
            self.act(agent_idx, action)

            # Temp
            reward = -1
            self.done = False

            # If agent passes over flag square, set to zero and mark agent as having the flag
            # if self.agent_position == self.flag_pos:
            #     self.has_flag = True
            #     reward = self.REWARD_CAPTURE
            #     self.flag_pos = None # remove flag from board otherwise the agent will spam the flag area
            # # Calculate rewards - game is done when the agent has the flag and reaches the capture position
            # elif self.has_flag and self.agent_position == self.capture_pos:
            #     reward = self.REWARD_GOAL
            #     self.done = True
            # else:
            #     reward = self.REWARD_STEP

        return self.grid, reward, self.done

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

        raise ValueError("Method not implemented")

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