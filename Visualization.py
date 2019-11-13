# Visualization in 3 dimensions for new simulation
# Author: Derek Knowles
# Date: 11/27/18


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from math import *
import time

class Visualization(QtCore.QThread):
    def __init__(self,result):
        self.result = result
        QtCore.QThread.__init__(self)


        self.scale = 2000.
        pg.setConfigOptions(antialias=True)             # set pyqtgraph options
        self.app = QtGui.QApplication([])               # create QT application
        self.w = gl.GLViewWidget()                      # create view widget
        self.w.setWindowTitle('AA228')                  # set title of window
        self.w.opts['distance'] = self.scale            # camera view position
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(self.scale/20, self.scale/20, self.scale/20) # set the size of the grid (distance between each line)
        self.w.addItem(grid) # add grid to viewer
        self.w.show()                                   # show the window
        self.w.setBackgroundColor('k')                  # set background color, option
        self.w.raise_()                             # bring window to the front
        self.step = 0
        self.total_steps = self.result.shape[1]

        self.robot_height = 100.0
        # add all three axis
        xaxis_pts = np.array([[0.0,0.0,0.0],            # north axis start and end point
                        [1.1*self.scale,0.0,0.0]])
        xaxis = gl.GLLinePlotItem(pos=xaxis_pts,color=pg.glColor('r'),width=3.0)   # create line plot item
        self.w.addItem(xaxis)                           # add item to graph
        yaxis_pts = np.array([[0.0,0.0,0.0],            # east axis start and end point
                        [0.0,1.1*self.scale,0.0]])
        yaxis = gl.GLLinePlotItem(pos=yaxis_pts,color=pg.glColor('g'),width=3.0)   # create line plot item
        self.w.addItem(yaxis)                           # add item to graph
        zaxis_pts = np.array([[0.0,0.0,0.0],            # down axis start and end point
                        [0.0,0.0,1.1*self.scale]])
        zaxis = gl.GLLinePlotItem(pos=zaxis_pts,color=pg.glColor('b'),width=3.0)   # create line plot item
        self.w.addItem(zaxis)                           # add item to graph

        # Ownship
        self.own_items = 21 # number of meshes that make up ownship
        self.own_3d = np.empty(self.own_items,dtype=object)
        cyl_object = gl.MeshData.cylinder(rows=100,cols=100,radius=[25.0,25.0],length=self.robot_height)   # cylinder object for each arm
        cyl_color = pg.glColor('g')
        self.own_3d[0] = gl.GLMeshItem(meshdata=cyl_object, smooth=True, drawFaces=True, drawEdges=False, color=cyl_color)
        self.w.addItem(self.own_3d[0])
        self.own_3d[0].translate(self.result[0,0],self.result[1,0],self.result[2,0])


        # add radial lines
        for j in range(0,20):
            angle = j*2*np.pi/20.0                      # angle for the radial line
            rad_line_pts = np.array([[0.0,0.0,0.0],     # start and end point for line
                                     [1.1*self.scale*np.cos(angle),1.1*self.scale*np.sin(angle),0.0]])
            self.own_3d[j+1] = gl.GLLinePlotItem(pos=rad_line_pts,color=pg.glColor('c'),width=0.1,antialias=True)   # create line plot item

            self.w.addItem(self.own_3d[j+1])
            self.own_3d[j+1].translate(self.result[0,0],self.result[1,0],self.result[2,0]+self.robot_height)


    def update(self):
        if self.step < self.total_steps-1:
            self.step += 1
            own_dx = self.result[0,self.step] - self.result[0,self.step-1]
            own_dy = self.result[1,self.step] - self.result[1,self.step-1]
            self.own_theta = degrees(atan2(own_dy,own_dx))
            for k in range(self.own_items):
                self.own_3d[k].translate(-self.result[0,self.step-1],
                                         -self.result[1,self.step-1],
                                         -self.result[2,self.step-1])
                #self.own_3d[k].rotate(self.own_theta,0,0,1)
                self.own_3d[k].translate(self.result[0,self.step],
                                         self.result[1,self.step],
                                         self.result[2,self.step])
        else:
            for ii in range(self.own_items):
                self.own_3d[ii].translate(self.result[0,0]-self.result[0,self.step],
                                          self.result[1,0]-self.result[1,self.step],
                                          self.result[2,0]-self.result[2,self.step])
            self.step = 0
            self.own_theta

        self.app.processEvents()
        time.sleep(0.05)



if __name__ == '__main__':
    pass
