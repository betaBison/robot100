#!/usr/bin/env python
"""
Author(s):  AA228 Group 100
Date:       Nov 12, 2019
Desc:       AA228 project
"""

from montecarlo import MonteCarlo
from visualization import Visualization as viz

def main():
    world_size = (20,20)       # size of box world (width,height)
    percent_obstacles = 0.1    # percent of spaces that are obstacles
    agent_vision_depth = 2     # raidus of square within agent can see obstacles

    # initializations
    graph = viz(world_size,percent_obstacles,agent_vision_depth)
    mc = MonteCarlo(graph.goal,graph.agent_origin,graph.obstacles)

    while(True):
        action = mc.compute()       # this currently does nothing
        state = mc.dummy_state()    # dummy state for viz tests
        graph.update(state)         # update visualization according to state

if __name__ == '__main__':
    main()
