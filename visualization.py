"""
Author(s):  D. Knowles
Date:       Nov 12, 2019
Desc:       3D visualization of grid world
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from math import *
import time

class Visualization(QtCore.QThread):
    def __init__(self,world_size,percent_obstacles,agent_vision_depth):
        """
        Desc: runs when instance of Visualization class is created

        Input(s):
            world_size: size of box world (width,height)
            percent_obstacles: percent of spaces that are occupied
            agent_vision_depth: raidus of square within agent can see obstacles
        Output(s):
            none
        """
        self.world_size = world_size                # size of box world (width,height)
        self.world_delta = 1.                       # discretization size of box world
        self.percent_obstacles = percent_obstacles  # percent of spaces that are occupied
        self.offset = self.world_delta/2.           # place objects in middle of grid box
        self.agent_vision_depth = agent_vision_depth# raidus of square within agent can see obstacles

        # setup the graphics window
        QtCore.QThread.__init__(self)
        pg.setConfigOptions(antialias=True)             # set pyqtgraph options
        self.app = QtGui.QApplication([])               # create QT application
        self.w = gl.GLViewWidget()                      # create view widget
        self.w.setWindowTitle('AA228 Robot100')         # set title of window
        sg = QtWidgets.QDesktopWidget().availableGeometry()             # get native window geometry
        self.w.setGeometry(sg.width()/2.,0,sg.width()/2.,sg.height())   # place window on right half of screen
        self.w.setCameraPosition(distance=0.7*np.sqrt(self.world_size[0]**2+self.world_size[1]**2),
                            elevation=40, azimuth=-135) # set initial camera position
        self.w.show()                                   # show the window
        self.w.setBackgroundColor('k')                  # set background color, option
        self.w.raise_()                                 # bring window to the front

        # setup grid
        grid = gl.GLGridItem()                                  # make a grid to represent the ground
        grid.setSize(x=world_size[0],y=world_size[1])
        grid.setSpacing(x=self.world_delta, y=self.world_delta)
        grid.translate(world_size[0]/2.,world_size[1]/2.,0.)    # translate so that it starts at (0,0)
        self.w.addItem(grid)                                    # add grid to viewer

        # add boundaries
        bound_mesh,bound_mesh_colors = self.get_boundary()      # create boundary mesh
        boundary = gl.GLMeshItem(vertexes=bound_mesh,           # defines the triangular mesh (Nx3x3)
                                 vertexColors=bound_mesh_colors,# defines mesh colors (Nx1)
                                 drawEdges=False,               # draw edges between mesh elements
                                 smooth=False)                  # speeds up rendering
                                 #computeNormals=False)         # speeds up rendering
        boundary.setGLOptions('additive')                       # allows them to be semi-transparent
        self.w.addItem(boundary)                                # add boundary to the viewer

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

        # generate obstacles
        self.generate_obstacles()                       # configure obstacle location
        self.drawObstacles(self.obstacles)              # draw obstacles in the viewer

        self.goal = self.random_location()              # select random goal
        self.agent_origin = self.random_location()      # select random origin
        self.agent_position = self.agent_origin         # set current position as origin

        # Add Agent to viewer
        cyl_height = 2.*self.world_delta
        cyl_radius = 0.8*(self.world_delta/2.)
        cyl_object = gl.MeshData.cylinder(rows=100,cols=100,radius=[cyl_radius,cyl_radius],length=cyl_height)   # cylinder object for each arm
        cyl_color = pg.glColor('r')
        self.own_3d = gl.GLMeshItem(meshdata=cyl_object, smooth=True, drawFaces=True, drawEdges=False, color=cyl_color)
        self.own_3d.translate(self.agent_origin[0]+self.offset,
                              self.agent_origin[1]+self.offset,
                              self.agent_origin[2])
        self.w.addItem(self.own_3d)

        # Add Goal to the viewer
        cyl_height = 2.*self.world_delta
        cyl_radius = 0.8*(self.world_delta/2.)
        cyl_object = gl.MeshData.cylinder(rows=100,cols=100,radius=[cyl_radius,cyl_radius],length=cyl_height)   # cylinder object for each arm
        cyl_color = pg.glColor('g')
        self.goal_3d = gl.GLMeshItem(meshdata=cyl_object, smooth=True, drawFaces=True, drawEdges=False, color=cyl_color)
        self.goal_3d.translate(self.goal[0]+self.offset,
                               self.goal[1]+self.offset,
                               self.goal[2])
        self.w.addItem(self.goal_3d)

    def get_boundary(self):
        """
        Desc: creates the boundary mesh

        Input(s):
            none
        Output(s):
            bound_mesh: mesh of boundary
            bound_mesh_colors: boundary mesh colors
        """
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

    def generate_obstacles(self):
        """
        Desc: chooses the spots where obstalces are located

        Input(s):
            none
        Output(s):
            none
        """
        spots = np.zeros((self.world_size[0],self.world_size[1],2))
        spots[:,:,0] = np.arange(self.world_size[0])
        spots[:,:,1] = np.arange(self.world_size[1]).reshape((self.world_size[1],1))
        self.spots = spots.reshape((self.world_size[0]*self.world_size[1],2))
        np.random.shuffle(self.spots)
        num_spots = self.spots.shape[0]
        num_obstacles = int(ceil(self.percent_obstacles*num_spots))
        self.obstacles = self.spots[:num_obstacles]
        self.spots = np.delete(self.spots,np.s_[:num_obstacles],0)

    def drawObstacles(self, map):
        """
        Desc: draws the obstacles in the viewer

        Input(s):
            map: array of obstacle locations
        Output(s):
            none
        """
        # draw map of the world: buildings
        self.fullMesh = np.array([], dtype=np.float32).reshape(0,3,3)
        self.fullMeshcolors = np.array([], dtype=np.float32).reshape(0,3,4)
        for ii in range(len(map)):
            mesh, meshColors = self.buildingVertFace(map[ii][0]+self.offset,
                                                     map[ii][1]+self.offset)
            self.fullMesh = np.concatenate((self.fullMesh, mesh), axis=0)
            self.fullMeshcolors = np.concatenate((self.fullMeshcolors, meshColors), axis=0)
        self.map = gl.GLMeshItem(vertexes= self.fullMesh,  # defines the triangular mesh (Nx3x3)
                      vertexColors= self.fullMeshcolors,  # defines mesh colors (Nx1)
                      drawEdges=False,  # draw edges between mesh elements
                      smooth=False,  # speeds up rendering
                      computeNormals=False)  # speeds up rendering
        self.w.addItem(self.map)


    def buildingVertFace(self, e, n):
        """
        Desc: returns the mesh for an individual obstacles

        Input(s):
            e: x location
            n: y location
        Output(s):
            none
        """
        width = self.world_delta
        height = 1.*self.world_delta*np.random.rand() + 0.5*self.world_delta
        # define patches for a building located at (x, y)
        # vertices of the building
        points = np.array([[e + width / 2, n + width / 2, 0], #NE 0
                         [e + width / 2, n - width / 2, 0],   #SE 1
                         [e - width / 2, n - width / 2, 0],   #SW 2
                         [e - width / 2, n + width / 2, 0],   #NW 3
                         [e + width / 2, n + width / 2, height], #NE Higher 4
                         [e + width / 2, n - width / 2, height], #SE Higher 5
                         [e - width / 2, n - width / 2, height], #SW Higher 6
                         [e - width / 2, n + width / 2, height]]) #NW Higher 7
        mesh = np.array([[points[0], points[3], points[4]],  #North Wall
                         [points[7], points[3], points[4]],  #North Wall
                         [points[0], points[1], points[5]],  # East Wall
                         [points[0], points[4], points[5]],  # East Wall
                         [points[1], points[2], points[6]],  # South Wall
                         [points[1], points[5], points[6]],  # South Wall
                         [points[3], points[2], points[6]],  # West Wall
                         [points[3], points[7], points[6]],  # West Wall
                         [points[4], points[7], points[5]],  # Top
                         [points[7], points[5], points[6]]])  # Top

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        grey = np.array([0.6, 0.6, 0.6, 1])
        light_grey = np.array([0.9, 0.9, 0.9, 1])
        meshColors = np.empty((10, 3, 4), dtype=np.float32)
        meshColors[0] = grey
        meshColors[1] = grey
        meshColors[2] = grey
        meshColors[3] = grey
        meshColors[4] = grey
        meshColors[5] = grey
        meshColors[6] = grey
        meshColors[7] = grey
        meshColors[8] = light_grey
        meshColors[9] = light_grey
        return mesh, meshColors

    def random_location(self):
        """
        Desc: choose random location. After a spot is used for an obstacle,
            goal, agent, etc., it is removed from the possible spots to
            remove the chance of objects being overlaid ontop of each other

        Input(s):
            none
        Output(s):
            x_location: random x location
            y_location: random y location
            z_location: random z location
        """
        if len(self.spots) < 1:
            print("no more spots left")
        x_location = self.spots[0][0]
        y_location = self.spots[0][1]
        z_location = 0
        self.spots = np.delete(self.spots,0,0)
        return [x_location,y_location,z_location]


    def update(self,state):
        """
        Desc: updates visualization

        Input(s):
            state: array [x,y,z] of agent's current position in 3D
        Output(s):
            none
        """
        self.own_3d.translate(-self.agent_position[0],
                              -self.agent_position[1],
                              -self.agent_position[2])
        self.own_3d.translate(state[0],
                              state[1],
                              state[2])
        self.agent_position = state

        self.update_immediate_obstacles()

        self.app.processEvents()

    def update_immediate_obstacles(self):
        """
        Desc: updates the color of the immediate obstacles

        Input(s):
            none
        Output(s):
            none
        """
        immediate_obstacles_indexes = np.where((abs(self.obstacles[:,1]-self.agent_position[1])<=self.agent_vision_depth) &
                                               (abs(self.obstacles[:,0]-self.agent_position[0])<=self.agent_vision_depth))[0]
        yellow = np.array([1., 1., 0., 1])
        dark_yellow = np.array([0.7, 0.7, 0., 1])
        grey = np.array([0.6, 0.6, 0.6, 1])
        light_grey = np.array([0.9, 0.9, 0.9, 1])

        for ii in range(int(self.fullMeshcolors.shape[0]/10)):
            meshColors = np.empty((10, 3, 4), dtype=np.float32)
            if ii in immediate_obstacles_indexes:
                meshColors[0] = yellow
                meshColors[1] = yellow
                meshColors[2] = yellow
                meshColors[3] = yellow
                meshColors[4] = yellow
                meshColors[5] = yellow
                meshColors[6] = yellow
                meshColors[7] = yellow
                meshColors[8] = dark_yellow
                meshColors[9] = dark_yellow
            else:
                meshColors[0] = grey
                meshColors[1] = grey
                meshColors[2] = grey
                meshColors[3] = grey
                meshColors[4] = grey
                meshColors[5] = grey
                meshColors[6] = grey
                meshColors[7] = grey
                meshColors[8] = light_grey
                meshColors[9] = light_grey
            self.fullMeshcolors[ii*10:10*(ii+1),:,:] = meshColors

        self.map = gl.GLMeshItem(vertexes= self.fullMesh,  # defines the triangular mesh (Nx3x3)
                      vertexColors= self.fullMeshcolors,  # defines mesh colors (Nx1)
                      drawEdges=False,  # draw edges between mesh elements
                      smooth=False,  # speeds up rendering
                      computeNormals=False)  # speeds up rendering




    def update_old(self):
        """
        Desc: outdated update function if given entire trajectory at once

        Input(s):
            none
        Output(s):
            none
        """
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
