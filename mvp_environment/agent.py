import numpy as np

class MVPAgent:
    def __init__(self, 
            actor_network, 
            critic_network,
            available_actions=np.array[1, 1, 1, 1, 1]):

        self.has_flag = False
        # up, down, left, right, nothing
        self.available_actions = available_actions
        self.actor_network = actor_network
        self.critic_network = critic_network

    def choose_action(self, state):
        """
        Choose an action.
        """
        action = None
        # apply mask based on agents abilities
        action *= self.available_actions
        return None

    def update_network(self):
        return None

    def save_models(self):
        return None
    
    def load_models(self):
        return None



    
