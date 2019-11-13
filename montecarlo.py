import numpy as np

class MonteCarlo():
    """
    """
    def __init__(self,goal,agent_origin):
        """
        Initializes monte carlo array

        Input(s)
        """
        self.goal = goal
        self.agent_origin = agent_origin
        self.dummy_state_setup() # for visualization testing


    def compute(self):
        """
        Desc: Compute monte carlo action

        Input(s):
            stuff
        Output(s):
            state: array (3x1) of agent's current position in 3D
        """
        state = self.dummy_state()

        return state


    def dummy_state_setup(self):
        """
        Desc: creates states that make the robot go towards the goal

        Input(s):
            none
        Output(s):
            none
        """
        num = 300.
        high = 10
        dt = np.linspace(0,high,int(num))
        result_x = 500.*np.sin(dt)
        result_y = 500.*(dt/high)*np.cos(dt)
        result_z = np.zeros(int(num))
        self.result = np.vstack((result_x,result_y,result_z))
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
        return state


        return state
