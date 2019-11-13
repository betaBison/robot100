"""
Author(s):  D. Knowles
Date:       Nov 12, 2019
Desc:       AA228 project
"""

from function import Function
from Visualization import Visualization as viz

# compute result
my_fun = Function()
result = my_fun.compute()

graph = viz(result) # visualize the simulation
while(True):
    #TODO:
    # action = monte_carlo(state,surrounds)
    # new state = update_state(action)
    
    graph.update() # update for each time step
