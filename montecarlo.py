"""
Author(s):  AA228 Group 100
Date:       Nov 12, 2019
Desc:       Monte Carlo Search
"""

import time
import logging
import random
# from threading import Thread

import numpy as np

class MonteCarlo():
    """
    """
    def __init__(self, gridworld, mode):
        """
        Initializes monte carlo array

        Input(s)
        mode:   0 - direct
                1 - random
                2 - monte carlo
        """
        # Thread.__init__(self)
        self.gridworld = gridworld
        self.mode = mode

    def run(self):
        """
        Desc: Compute monte carlo action

        Input(s):

        Output(s):
            action
        """
        while True:
            actions = []
            if self.mode == 0:
                for agent in self.gridworld.agents:
                    actions.append(0)
                time.sleep(0.1)
            elif self.mode == 1:
                for agent in self.gridworld.agents:
                    dist = np.inf
                    nearest_goal = None
                    for goal in self.gridworld.goals:
                        if (((goal.pos[0]-agent.pos[0])**2
                            + (goal.pos[1]-agent.pos[1])**2)**0.5 < dist):
                            nearest_goal = goal
                    dist = nearest_goal.pos - agent.pos
                    logging.debug("Dist to goal: (%d, %d)" % (dist[0], dist[1]))
                    if abs(dist[0]) > abs(dist[1]):
                        action = 0 if dist[0] < 0 else 1
                    else:
                        action = 2 if dist[1] > 0 else 3
                    if (dist[0] == 0 and dist[1] == 0):
                        action = None
                    actions.append(action)
                time.sleep(0.1)
            elif self.mode == 2:
                for agent in self.gridworld.agents:
                    actions.append(random.randint(0, 3))
                time.sleep(0.1)

            self.gridworld.update(actions)
