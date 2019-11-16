"""
Author(s):  AA228 Group 100
Date:       Nov 15, 2019
Desc:       Goal
"""

import numpy as np

class Goal():
    def __init__(self, pos, reward):
        self.pos = np.array(pos, dtype=int)
        self.reward = reward