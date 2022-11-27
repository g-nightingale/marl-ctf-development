import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
import time

class GridworldCtf:
    """
    A GridWorld capture the flag environment.
    
    """

    def __init__(self, game_mode='static') -> None:
        self.GAME_MODE = game_mode # static or random
        self.BLOCK_NUM = 1
        self.GRID_LEN = 10
        self.REWARD_GOAL = 100
        self.REWARD_STEP = -1
        self.REWARD_CAPTURE = -1
        self.reset()

    def reset(self):
        """ 
        Reset the environment. 
        """
        self.grid = np.flip(np.loadtxt("./map.txt", dtype=np.int8), axis = 0)

        if self.GAME_MODE=='static':
            self.agent_position = (8, 8)
            self.agent_start_pos = self.agent_position
            self.flag_pos = (1, 8)
            self.capture_pos = (8, 1)
        elif self.GAME_MODE =='random':
            self.agent_position = (np.random.randint(7, 10), np.random.randint(10))
            self.agent_start_pos = self.agent_position
            self.flag_pos = (np.random.randint(0, 3), np.random.randint(10))
            capture_pos = None
            while capture_pos is None or capture_pos == self.agent_position:
                capture_pos = (np.random.randint(7, 10), np.random.randint(10))
            self.capture_pos = capture_pos
        else:
            raise ValueError("Game mode must be set to either 'static' or 'random'")

        self.init_objects()
        self.has_flag = False
        self.done = False

    def init_objects(self) -> np.array:
        """"
        Initialise the grid.
        """

        # Add agent
        self.grid[self.agent_start_pos] = 2

        # Add target capture zone
        self.grid[self.capture_pos] = 3

        # Add flag
        self.grid[self.flag_pos] = 4


    def move(self, action) -> None:
        """
        Move the agent.
        """

        # Move up
        if action == 0:
            if self.agent_position[0] > 0 and self.grid[self.agent_position[0] - 1, self.agent_position[1]] != self.BLOCK_NUM:
                self.grid[self.agent_position] = 0
                self.agent_position = (self.agent_position[0] - 1, self.agent_position[1])
                self.grid[self.agent_position] = 2
        # Move down
        if action == 1:
            if self.agent_position[0] < (self.GRID_LEN-1) and self.grid[self.agent_position[0] + 1, self.agent_position[1]] != self.BLOCK_NUM:
                self.grid[self.agent_position] = 0
                self.agent_position = (self.agent_position[0] + 1, self.agent_position[1])
                self.grid[self.agent_position] = 2
        # Move right
        if action == 2:
            if self.agent_position[1] < (self.GRID_LEN-1) and self.grid[self.agent_position[0], self.agent_position[1] + 1] != self.BLOCK_NUM:
                self.grid[self.agent_position] = 0
                self.agent_position = (self.agent_position[0], self.agent_position[1] + 1)
                self.grid[self.agent_position] = 2
        # Move left
        if action == 3:
            if self.agent_position[1] > 0 and self.grid[self.agent_position[0], self.agent_position[1] - 1] != self.BLOCK_NUM:
                self.grid[self.agent_position] = 0
                self.agent_position = (self.agent_position[0], self.agent_position[1] - 1)
                self.grid[self.agent_position] = 2
       
    def step(self, action) -> tuple:
        """
        Take a step in the environment.
        """

        # Move the agent
        self.move(action)

        # If agent passes over flag square, set to zero and mark agent as having the flag
        if self.agent_position == self.flag_pos:
            self.has_flag = True
            reward = self.REWARD_CAPTURE
            self.flag_pos = None # remove flag from board otherwise the agent will spam the flag area
        # Calculate rewards - game is done when the agent has the flag and reaches the capture position
        elif self.has_flag and self.agent_position == self.capture_pos:
            reward = self.REWARD_GOAL
            self.done = True
        else:
            reward = self.REWARD_STEP

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
        #env_plot[self.position] = 4
        #env_plot = np.flip(env_plot, axis = 0)

        # Plot the gridworld.
        c = ["lightgrey", "black", "blue", "lightblue", "red"]
        cmap = colors.ListedColormap(c)
        bounds = list(range(len(c)))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(env_plot, cmap=cmap, norm=norm, zorder=0)

        # Set up axes.
        ax.grid(which='major', axis='both', linestyle='-', color='0.4', linewidth=2, zorder=1)
        ax.set_xticks(np.arange(-0.5, self.grid.shape[1] , 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1))
        ax.set_yticklabels([])

        # Draw everything.
        #fig.canvas.draw()
        #fig.canvas.flush_events()

        plt.show()

        # Sleep if desired.
        if (sleep_time > 0) :
            time.sleep(sleep_time)

    def play(self):
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

        while True:  
            print(f"Move {move_counter}")
            self.render()

            if move_counter > 1:
                print(f"reward: {reward}, done: {done}")

            while raw_action not in ['w', 's', 'a', 'd', 'x']:
                raw_action = input("Enter an action")

            if raw_action == 'x':
                print(f"Game exited")
                break

            action = ACTIONS_MAP[raw_action]

            _, reward, done = self.step(int(action))

            total_score += reward
            move_counter += 1

            raw_action = None

            if done:
                print(f"You win!, total score {total_score}")
                break