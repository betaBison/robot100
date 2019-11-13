#!/usr/bin/env python
"""
Author(s):  D. Knowles
Date:       Nov 12, 2019
Desc:       AA228 project
"""

from montecarlo import MonteCarlo
from Visualization import Visualization as viz

def main():
    # initial parameters
    world_size = (100.,100.) # size of box world
    world_delta = 10. # discretization size of box world

    # initializations
    graph = viz(world_size,world_delta) # visualize the simulation
    mc = MonteCarlo(graph.goal,graph.agent_origin)

    while(True):
        #TODO:
        # action = monte_carlo(state,surrounds)
        # new state = update_state(action)
        state = mc.compute()
        graph.update(state) # update for each time step



if __name__ == '__main__':
    main()
