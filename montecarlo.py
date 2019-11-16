"""
Author(s):  AA228 Group 100
Date:       Nov 12, 2019
Desc:       Monte Carlo Search
"""

import time
import logging
import random

import numpy as np

class MonteCarlo():
    """
    """
    def __init__(self, gridworld, mode=0):
        """
        Initializes monte carlo array

        Input(s)
        mode:   0 - direct
                1 - random
                2 - monte carlo
        """
        self.gridworld = gridworld
        self.mode = mode

    def compute(self):
        """
        Desc: Compute monte carlo action

        Input(s):

        Output(s):
            action
        """
        actions = []
        if self.mode == 0:
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
        elif self.mode == 1:
            for agent in self.gridworld.agents:
                actions.append(random.randint(0, 3))
        return actions

    def constrain_to_state(self,input):
        """
        Desc: Compute monte carlo action

        Input(s):
            input: array [x,y,z] of agent's current position in 3D (floats)
        Output(s):
            state: array [x,y,z] of agent's current position in 3D (ints)
        """
        output = input.copy()
        output[0] = int(input[0])
        output[1] = int(input[1])
        output[2] = int(input[2])
        return output


    def dummy_state_setup(self):
        """
        Desc: creates states that make the robot go towards the goal

        Input(s):
            none
        Output(s):
            none
        """
        num = 100
        dx = np.linspace(self.agent_origin[0],self.goal[0],num)
        dy = np.linspace(self.agent_origin[1],self.goal[1],num)
        dz = np.linspace(self.agent_origin[2],self.goal[2],num)
        # result_x = 500.*np.sin(dt)
        # result_y = 500.*(dt/high)*np.cos(dt)
        # result_z = np.zeros(int(num))
        self.result = np.vstack((dx,dy,dz))
        self.step = 0

    def dummy_state(self):
        """
        Desc: Compute monte carlo action

        Input(s):
            none
        Output(s):
            state: array (3x1) of agent's current position in 3D
        """

        state = self.result[:,self.step]
        if self.step < self.result.shape[1]-1:
            self.step += 1
        else:
            self.step = 0
        time.sleep(0.05)
        return self.constrain_to_state(state)
