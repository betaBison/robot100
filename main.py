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

def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG)
    # initializations
    gridworld = GridWorld(args.size, args.interval, args.obstacles, args.vision, args.phase)
    logging.info("Generated grid world!")
    viz = Visualization(gridworld)
    logging.info("Visuals created")
    mc = MonteCarlo(gridworld, mode=args.method)
    logging.info("Initialized Monte Carlo method")
    
    mc.start()
    viz.start()
    viz.app.exec_()
        

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
    parser.add_argument('-m', '--method', default=2, type=int,
                        help='Method to adopt to solve. Modes: 0 - Monte Carlo, 1 - Direct, 2 - Random. Default is 2 (in beta)')
    parser.add_argument('-p', '--phase', action='store_true',
                        help='Whether or not agent can phase through obstacles. Default is off.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose mode on or off. Default is off.")
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Debug mode on or off. Default is off.")
    args = parser.parse_args()
    main(args)
