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
    def __init__(self,world_size,world_delta):
        self.world_size = world_size
        self.world_delta = world_delta

        QtCore.QThread.__init__(self)
        pg.setConfigOptions(antialias=True)             # set pyqtgraph options
        self.app = QtGui.QApplication([])               # create QT application
        self.w = gl.GLViewWidget()                      # create view widget
        self.w.setWindowTitle('AA228')                  # set title of window
        self.w.opts['distance'] = 2.*world_size[0]            # camera view position
        self.w.show()                                   # show the window
        self.w.setBackgroundColor('k')                  # set background color, option
        self.w.raise_()                             # bring window to the front



        # setup grid
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.setSize(x=world_size[0],y=world_size[1])
        grid.setSpacing(x=world_delta, y=world_delta)
        grid.translate(world_size[0]/2.,world_size[1]/2.,0.)
        self.w.addItem(grid) # add grid to viewer

        # add boundaries
        bound_mesh,bound_mesh_colors = self.get_boundary()
        boundary = gl.GLMeshItem(vertexes=bound_mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=bound_mesh_colors, # defines mesh colors (Nx1)
            drawEdges=False,  # draw edges between mesh elements
            smooth=False)  # speeds up rendering
            #computeNormals=False)  # speeds up rendering
        boundary.setGLOptions('additive')
        self.w.addItem(boundary)

        # add all three colored axes
        xaxis_pts = np.array([[0.0,0.0,0.0],            # north axis start and end point
                        [1.1*self.world_size[0],0.0,0.0]])
        xaxis = gl.GLLinePlotItem(pos=xaxis_pts,color=pg.glColor('r'),width=3.0)   # create line plot item
        self.w.addItem(xaxis)                           # add item to graph
        yaxis_pts = np.array([[0.0,0.0,0.0],            # east axis start and end point
                        [0.0,1.1*self.world_size[1],0.0]])
        yaxis = gl.GLLinePlotItem(pos=yaxis_pts,color=pg.glColor('g'),width=3.0)   # create line plot item
        self.w.addItem(yaxis)                           # add item to graph
        zaxis_pts = np.array([[0.0,0.0,0.0],            # down axis start and end point
                        [0.0,0.0,5.0*self.world_delta]])
        zaxis = gl.GLLinePlotItem(pos=zaxis_pts,color=pg.glColor('b'),width=3.0)   # create line plot item
        self.w.addItem(zaxis)                           # add item to graph



        self.goal = [10.,10.,0.]
        self.agent_origin = [90.,90.,0.]

        self.robot_height = 100.0


        # Agent
        cyl_height = 2.*world_delta
        cyl_radius = 0.8*(world_delta/2.)
        cyl_object = gl.MeshData.cylinder(rows=100,cols=100,radius=[cyl_radius,cyl_radius],length=cyl_height)   # cylinder object for each arm
        cyl_color = pg.glColor('r')
        self.own_3d = gl.GLMeshItem(meshdata=cyl_object, smooth=True, drawFaces=True, drawEdges=False, color=cyl_color)
        self.own_3d.translate(self.agent_origin[0],self.agent_origin[1],self.agent_origin[2])
        self.w.addItem(self.own_3d)

    def get_boundary(self):
        h = 1.*self.world_delta # height of boundary
        d = 3.*self.world_delta # depth of boundary
        points = np.array([[0.0,0.0,0.0], # 0
                           [0.0,self.world_size[1],0.0],
                           [self.world_size[0],self.world_size[1],0.0],
                           [self.world_size[0],0.0,0.0],
                           [-d,-d,0.0],
                           [-d,self.world_size[1]+d,0.0],
                           [self.world_size[0]+d,self.world_size[1]+d,0.0],
                           [self.world_size[0]+d,-d,0.0],

                           [0.0,0.0,h], # 8
                           [0.0,self.world_size[1],h],
                           [self.world_size[0],self.world_size[1],h],
                           [self.world_size[0],0.0,h],
                           [-d,-d,h], # 12
                           [-d,self.world_size[1]+d,h],
                           [self.world_size[0]+d,self.world_size[1]+d,h],
                           [self.world_size[0]+d,-d,h]]) # 15
        bound_mesh = np.array([[points[8], points[12], points[13]],
                               [points[8], points[9], points[13]],
                               [points[9], points[13], points[14]],
                               [points[9], points[10], points[14]],
                               [points[10], points[14], points[15]],
                               [points[10], points[11], points[15]],
                               [points[11], points[12], points[15]],
                               [points[8], points[11], points[12]],
                               [points[0], points[4], points[5]], # start of bottom
                               [points[0], points[1], points[5]],
                               [points[1], points[5], points[6]],
                               [points[1], points[2], points[6]],
                               [points[2], points[6], points[7]],
                               [points[2], points[3], points[7]],
                               [points[3], points[4], points[7]],
                               [points[0], points[3], points[4]],
                               [points[8], points[9], points[0]], # start of inside
                               [points[0], points[1], points[9]],
                               [points[1], points[9], points[10]],
                               [points[1], points[2], points[10]],
                               [points[2], points[10], points[11]],
                               [points[2], points[3], points[11]],
                               [points[3], points[11], points[8]],
                               [points[3], points[0], points[8]],
                               [points[4], points[12], points[13]], # start of outside
                               [points[4], points[5], points[13]],
                               [points[5], points[13], points[14]],
                               [points[5], points[6], points[14]],
                               [points[6], points[14], points[15]],
                               [points[6], points[7], points[15]],
                               [points[7], points[15], points[12]],
                               [points[7], points[4], points[12]]])

        bound_mesh_colors = np.zeros((32, 3, 4))
        bound_mesh_colors[:,:,0] = 0.7
        bound_mesh_colors[:,:,1] = 0.7
        bound_mesh_colors[:,:,2] = 0.7
        bound_mesh_colors[:,:,3] = 0.4

        return bound_mesh,bound_mesh_colors

    def update(self,state):
        self.app.processEvents()
        time.sleep(0.05)



    def update_old(self):
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





if __name__ == '__main__':
    pass
