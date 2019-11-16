#!/usr/bin/env python
"""
Author(s):  AA228 Group 100
Date:       Nov 12, 2019
Desc:       AA228 project
"""

import logging
import argparse
import time

from gridworld import GridWorld
from visualization import Visualization
from montecarlo import MonteCarlo
from obstacles import Obstacle
from agent import Agent
from obstacles import Obstacle

def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG)
    print(args)
    # initializations
    gridworld = GridWorld(args.size, args.interval, args.obstacles, args.vision)
    logging.info("Generated grid world!")
    viz = Visualization(gridworld)
    logging.info("Visuals created")
    mc = MonteCarlo(gridworld)
    logging.info("Initialized Monte Carlo method")

    while(True):
        actions = mc.compute()       # this currently does nothing
        gridworld.update(actions)
        viz.update()
        time.sleep(0.01)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AA228 Robot100")
    parser.add_argument('-s', '--size', default=(20,20), nargs='+', type=int,
                        help='Grid world size. Default is 20x20')
    parser.add_argument('-i', '--interval', default=1, type=float,
                        help='Grid world discretization interval. Default is 1')
    parser.add_argument('-o', '--obstacles', default=0.1, type=float,
                        help='Percentage of obstacles in world. Default is 0.1')
    parser.add_argument('-vi', '--vision', default=2, type=float,
                        help='Depth of vision of agent. Default is 2')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose mode on or off. Default is off.")
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Debug mode on or off. Default is off.")
    args = parser.parse_args()
    main(args)
