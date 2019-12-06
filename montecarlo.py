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
    def __init__(self, gridworld, mode):
        """
        Initializes monte carlo array

        Input(s)
        mode:   0 - direct
                1 - random
                2 - monte carlo
        """
        self.gridworld = gridworld
        self.mode = mode

        self.T = [] # set of visited states
        self.Q = [] # value function for each state,action pair
        self.N = [] # number of times we have taken an action from a state

        for agent in self.gridworld.agents:
            self.T.append([])
            self.Q.append(np.zeros((self.gridworld.world_size[0],
                                    self.gridworld.world_size[1],
                                    len(agent.possible_actions))))
            self.N.append(np.zeros((self.gridworld.world_size[0],
                                    self.gridworld.world_size[1],
                                    len(agent.possible_actions))))

        self.mc_trees = []          # state points for visualization
        self.counter = 0

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
                # DMU book Algorithm 4.9 Monte Carlo tree search
                for agent_num in range(len(self.gridworld.agents)):
                    self.mc_trees = []
                    state_copy = self.gridworld.agents[agent_num].pos.copy()
                    new_action = self.mc_select_action(agent_num,state_copy,200)
                    actions.append(new_action)
                    mc_trees_object = np.asarray(self.mc_trees)
                    self.gridworld.visualization.update_mc_trees(mc_trees_object)
            elif self.mode == 1:
                for agent in self.gridworld.agents:
                     action = self.direct_to_goal(agent)
                     actions.append(action)
                time.sleep(0.1)
            elif self.mode == 2:
                for agent in self.gridworld.agents:
                    actions.append(random.randint(0, 3))
                time.sleep(0.1)
            self.gridworld.update(actions)


    def direct_to_goal(self,agent):
        """
        Desc: returns action that gets it closest to the nearest goal
        Input(s):
            agent: class instance of agent
        Output(s):
            action: action to get to goal
        """
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
        return action

    def direct_to_goal_with_state(self,agent,s):
        """
        Desc: returns action that gets it closest to the nearest goal
        Input(s):
            agent: class instance of agent
            s: current_state
        Output(s):
            action: action to get to goal
        """
        dist = np.inf
        nearest_goal = None
        for goal in self.gridworld.goals:
            if (((goal.pos[0]-s[0])**2
                + (goal.pos[1]-s[1])**2)**0.5 < dist):
                nearest_goal = goal
        dist = nearest_goal.pos - s
        logging.debug("Dist to goal: (%d, %d)" % (dist[0], dist[1]))
        if abs(dist[0]) > abs(dist[1]):
            action = 0 if dist[0] < 0 else 1
        else:
            action = 2 if dist[1] > 0 else 3
        if (dist[0] == 0 and dist[1] == 0):
            action = None
        return action

    def mc_select_action(self,agent_num,s,d):
        """
        Desc: Compute monte carlo action
        DMU book Algorithm 4.9 select action
        Input(s):
            agent_num: current agent number
            s: current state
            d: horizon (or depth)
        Output(s):
            best_action: action that maximizes value function Q (argmax Q(s,a))
        """
        time0 = time.time()
        loop_time = 2.0
        while (time.time() - time0) < loop_time:
            self.mc_simulate(agent_num,s,d)
        best_action = np.argmax(self.Q[agent_num][s[0],s[1],:])
        return best_action

    def mc_simulate(self,agent_num,s,d):
        """
        Desc: Compute monte carlo action
        DMU book Algorithm 4.9 simulate
        Input(s):
            agent_num: current agent number
            s: current state
            d: horizon (or depth)
        Output(s):
            q:
        """
        c = 2.0     # parameter that controls the amount of exploration
                    # in the search DMU book eq. 4.36
        gamma = 0.8 # discount factor

        if d == 0:
            # if depth is zero get out
            return 0.0

        sd = np.abs(s-self.gridworld.agents[agent_num].pos) # state difference
        if sd[0] <= self.gridworld.agent_vision_depth and sd[1] <= self.gridworld.agent_vision_depth:
            # near the current position use argmax
            a = self.argmaximize(agent_num,s,c)
        else:
            # far from the current position use direct to goal
            a = self.direct_to_goal_with_state(self.gridworld.agents[agent_num],s)

        # next state based on generative model
        s_prime = self.gridworld.agents[agent_num].generative_model(s.copy(),a)
        # next state shouldn't be outside of the arena
        s_prime = self.gridworld.conform_state_to_bounds(s_prime)
        # get the reward of going into the next state
        r = self.gridworld.check_reward(self.gridworld.agents[agent_num],s_prime)
        if r > 0.0:
            # ends simulation if it reaches goal
            result = 0.0
        else:
            # recursive simulation
            result = self.mc_simulate(agent_num,s_prime,d-1)
        q = r + gamma*result
        # increment the count for number of times visited
        self.N[agent_num][s[0],s[1],a] += 1
        # update the expected utility
        self.Q[agent_num][s[0],s[1],a] += (q-self.Q[agent_num][s[0],s[1],a])/self.N[agent_num][s[0],s[1],a]

        # add the state to the graph for visualization
        self.mc_trees.append([[s[0]+0.5,s[1]+0.5],[s_prime[0]+0.5,s_prime[1]+0.5]])

        # update visualization every 100 loops
        self.counter += 1
        if self.counter % 100 == 0:
            mc_trees_object = np.asarray(self.mc_trees)
            self.gridworld.visualization.update_mc_trees(mc_trees_object)

        return q


    def argmaximize(self,agent_num,s,c):
        """
        Desc: returns best action
        DMU book eq. 4.36
        Input(s):
            agent_num: agent number
            s: current state
            c: parameter that controls the amount of exploration
        Output(s):
            a: arg max of array
        """
        num_actions = len(self.gridworld.agents[agent_num].possible_actions)
        x = s[0]
        y = s[1]
        total = np.zeros(num_actions)
        for ii in range(num_actions):
            if self.N[agent_num][x,y,ii] == 0:
                total[ii] = np.inf
            else:
                total[ii] = self.Q[agent_num][x,y,ii] \
                         + c*np.sqrt(np.log(np.sum(self.N[agent_num][x,y,:])) \
                         / self.N[agent_num][x,y,ii])

        best_action = np.argmax(total)

        return best_action
