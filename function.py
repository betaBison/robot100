import numpy as np

class MonteCarlo():
    """
    """
    def __init__(self):
        pass

    def compute(self):
        num = 300.
        high = 10
        dt = np.linspace(0,high,int(num))
        result_x = 500.*np.sin(dt)
        result_y = 500.*(dt/high)*np.cos(dt)
        result_z = np.zeros(int(num))
        result = np.vstack((result_x,result_y,result_z))

        return result
