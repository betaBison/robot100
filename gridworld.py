"""
Author(s):  AA228 Group 100
Date:       Nov 15, 2019
Desc:       Obstacle
"""

#%%
import logging
from math import ceil
import random

import numpy as np

from visualization import Visualization
from agent import Agent
from obstacles import Obstacle
from goal import Goal

class GridWorld():
    def __init__(self, world_size, world_delta, obs_percent, agent_vision_depth, phase, rewards=10., penalty=-10., barrier_mode=0):
        """
        Desc: runs when instance of Visualization class is created

        Input(s):
            world_size: size of box world (width,height)
            world_delta: interval of world
            obs_percent: percent of spaces that are occupied
            agent_vision_depth: raidus of square within agent can see obstacles

            Optional-
            rewards: reward for getting to goal
            penalty: penalty for being in obstacle
        Output(s):
            none
        """
        self.world_size = world_size
        self.world_delta = world_delta      # used on virtualization
        self.obs_percent = obs_percent
        self.agent_vision_depth = agent_vision_depth
        self.spots = self.generate_spots()

        # initialize gridworld elements
        self.obstacle_indices, self.obstacles = self.generate_obstacles(penalty)
        self.goals = [self.generate_goal(rewards)]
        self.agents = [self.generate_agent(0, phase)]
        # # initialize visualization
        self.visualization = Visualization(self)

    def generate_spots(self):
        """
        Desc: Initialize state labels (0,0),(0,1),(0,2)
            ...(world_size[0],world_size[0])

        Input(s):
            none
        Output(s):
            spots:  Randomized list of coordinates for spots
        """
        spots = np.zeros((self.world_size[1], self.world_size[0], 3), dtype=int)
        spots[:,:,0] = np.array([list(range(self.world_size[0]))]*self.world_size[1])
        spots[:,:,1] = np.array([list(range(self.world_size[1]))]*self.world_size[0]).transpose()
        return spots

    def generate_obstacles(self, penalty):
        """
        Desc: Generates all obstacles with penalty. This is selected by
            picking the spots randomly. Also removes occupied spots.

        Input(s):
            penalty:    Penalty incurred for each obstacle
        Output(s):
            spots:      Randomized list of coordinates for spots
        """
        obstacles = []
        num_spots = self.world_size[0] * self.world_size[1]
        num_obstacles = int(ceil(self.obs_percent * num_spots))
        logging.info("Num of obstacles: %d" % num_obstacles)
        # choose obstacles
        obstacle_indices = np.unravel_index(random.sample(range(num_spots), num_obstacles), self.world_size)
        # set spot to obstacle
        obstacle_indices = np.array(list(zip(obstacle_indices[0], obstacle_indices[1])))
        # unravel_index gives y first
        self.spots[obstacle_indices[:,1], obstacle_indices[:,0], 2] = 3
        logging.info("Obstacle positions: ")
        for pos in obstacle_indices:
            logging.info(pos)
        obstacles = [Obstacle(pos, penalty) for pos in obstacle_indices]
        return obstacle_indices, obstacles

    def generate_goal(self, reward):
        """
        Desc: Generate goal on the gridworld. This is done by choosing a
            random spot.

        Input(s):
            reward: reward designated for this goal
        Output(s):
            none
        """
        initial_pos = self.choose_spot()
        logging.info("Goal pos: (%d, %d)" % (initial_pos[0], initial_pos[1]))
        self.spots[initial_pos[1], initial_pos[0], 2] = 2
        return Goal(initial_pos, reward)

    def generate_agent(self, initial_reward, phase):
        """
        Desc: Generate agent on the gridworld. This is done by choosing a
            random spot.

        Input(s):
            reward: reward designated for this goal
        Output(s):
            none
        """
        initial_pos = self.choose_spot()
        logging.info("Initial agent pos: (%d, %d)" % (initial_pos[0], initial_pos[1]))
        self.spots[initial_pos[1], initial_pos[0], 2] = 1
        return Agent(initial_pos, initial_reward, phase)

    def choose_spot(self):
        """
        Desc: Choose random location. It is randomized by the way
            spots are generated. So the next spot is selected.
            After a spot is used for an obstacle, goal, agent, etc., it
            is removed from the possible spots to remove the chance of
            objects being overlaid ontop of each other

        Input(s):
            none
        Output(s):
            x_location: random x location
            y_location: random y location
            z_location: random z location
        """
        available_spots = self.spots[np.where(self.spots[:,:,2] == 0)]
        if np.shape(available_spots)[0] == 0:
            logging.error("No more spots left")
            return None
        chosen_spot = available_spots[random.randint(0, np.shape(available_spots)[0] - 1)]
        return [chosen_spot[0], chosen_spot[1]]

    def update(self, actions):
        """
        Desc: Update gridworld based on computed actions.

        Input(s):
            actions:    actions computed
        Output(s):
            none
        """
        for i, agent in enumerate(self.agents):
            logging.debug('Agent %d: %d, %d, %d' % (i, agent.pos[0], agent.pos[1], agent.reward))
            next_pos = agent.next_state(actions[i])

            if not agent.phase:
                for obstacle_index in agent.obs:
                    if np.array_equal(next_pos, self.obstacles[obstacle_index].pos):
                        logging.debug('Agent %d was blocked by obstacle at (%d, %d)' % (i,
                                                                                self.obstacles[obstacle_index].pos[0],
                                                                                self.obstacles[obstacle_index].pos[1]))
                        actions[i] = None
            if ((next_pos[0] < 0) or
                (next_pos[0] >= self.world_size[0]) or
                (next_pos[1] < 0) or
                (next_pos[1] >= self.world_size[1])):
                # can add penalty for trying to leave world here
                logging.debug('Agent %d attempted to travel out of world' % i)
                actions[i] = None
            next_pos = agent.next_state(actions[i])
            agent.set_next_state(next_pos)
            agent.set_next_action(actions[i])
            nearby_obs = np.where(np.all(np.abs(self.obstacle_indices-agent.pos) <= self.agent_vision_depth, axis=1))[0]
            agent.observe(nearby_obs)
            current_reward = self.check_reward(agent,next_pos)
            agent.update_reward(current_reward)
        self.visualization.run()

    def conform_state_to_bounds(self,state):
        while state[0] < 0.0:
            state[0] += 1
        while state[0] >= self.world_size[0]:
            state[0] -= 1
        while state[1] < 0.0:
            state[1] += 1
        while state[1] >= self.world_size[1]:
            state[1] -= 1
        return state

    def check_reward(self,agent,state):
        """
        Desc: returns reward for taking an action at a state
        currently this is implemented only based on the new state
        Input(s):
            agent: agent class instance
            state: current agent position
        Output(s):
            reward: return reward
        """
        for goal in self.goals:
            if np.array_equal(state, goal.pos):
                return goal.reward
        if not agent.phase:
            for obstacle_index in agent.obs:
                if np.array_equal(state, self.obstacles[obstacle_index].pos):
                    return self.obstacles[obstacle_index].penalty
            return 0.0
        else:
            return 0.0
