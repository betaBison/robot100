"""
Author(s):  AA228 Group 100
Date:       Nov 15, 2019
Desc:       Agent
"""

import logging

import numpy as np

class Agent():
    def __init__(self, pos, reward, phase):
        self.pos = np.array(pos, dtype=int)    # agent's current position
        self.reward = reward        # agent's current reward
        self.obs = []               # agent's currently observable obstacles
        self.next_action = None
        self.phase = phase

    # Actions are:
    # 0 - left
    # 1 - right
    # 2 - up
    # 3 - down
    def next_state(self, action):
        pos = np.array(self.pos, copy=True)
        if (action == 0):
            pos[0] -= 1
        elif (action == 1):
            pos[0] += 1
        elif (action == 2): 
            pos[1] += 1
        elif (action == 3):
            pos[1] -= 1
        elif (action == None):
            pass
        else:
            logging.warning("Unrecognized action. Agent remaining still.")
        return pos
    
    def set_next_state(self, state):
        self.pos = state

    def set_next_action(self, action):
        self.next_action = action

    # update reward for agent
    # world should check for agent pos and reward/obs pos
    # then call this function to update reward of this agent
    def update_reward(self, reward):
        self.reward += reward

    # call this function to update the currently observable obstacles
    def observe(self, nearby_obs):
        self.obs = nearby_obs
