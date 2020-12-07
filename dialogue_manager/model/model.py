
import random
import math

class model():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.is_test = False
    
    def get_next_action(self, state):
        pass
    
    def update_model(self, dialogue_history, state, action, next_state, reward):
        pass
    
    def is_simulation(self):
        pass
    
    def set_test_phase(self, flag=False, change=True):
        self.is_test = flag
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass

    def plot_statistics(self):
        pass