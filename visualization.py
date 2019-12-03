"""
Author(s):  AA228 group
Date:       Nov 12, 2019
Desc:       3D visualization of grid world
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtGui import QVector3D
import time

RED                 = np.array([1., 0., 0., 1])
GREEN               = np.array([0., 1., 0., 1])
BLUE                = np.array([0., 0., 1., 1])
YELLOW              = np.array([1., 1., 0., 1])
DARK_YELLOW         = np.array([0.7, 0.7, 0., 1])
GREY                = np.array([0.6, 0.6, 0.6, 1])
LIGHT_GREY          = np.array([0.9, 0.9, 0.9, 1])
TRANSPARENT_GREY    = np.array([0.7, 0.7, 0.7, 0.4])

class Visualization():
    def __init__(self, gridworld, view_axis=False):
        """
        Desc: runs when instance of Visualization class is created

        Input(s):
            grid: Reference to grid object which represents gridworld
            view_axis: Whether or not to show axes
        Output(s):
            none
        """
        self.gridworld = gridworld
        self.offset = self.gridworld.world_delta/2.           # place objects in middle of grid box

        # setup the graphics window
        # QtCore.QThread.__init__(self)
        pg.setConfigOptions(antialias=True)             # set pyqtgraph options
        self.app = QtGui.QApplication([])               # create QT application
        self.w = gl.GLViewWidget()                      # create view widget
        self.w.setWindowTitle('AA228 Robot100')         # set title of window
        sg = QtWidgets.QDesktopWidget().availableGeometry()             # get native window geometry
        self.w.setGeometry(sg.width()/2.,sg.height()/3.,sg.width()/2.,sg.height()/2.)   # place window on right half of screen
        # self.w.setCameraPosition(distance=0.7*np.sqrt(self.gridworld.world_size[0]**2+self.gridworld.world_size[1]**2),
                            # elevation=40, azimuth=-135) # set initial camera position
        self.w.setCameraPosition(distance=0.7*np.sqrt(self.gridworld.world_size[0]**2+self.gridworld.world_size[1]**2),
                            elevation=90, azimuth=0)
        self.w.opts['center'] = QVector3D(self.gridworld.world_size[0]/2.,
                                          self.gridworld.world_size[1]/2.,
                                          np.sqrt(self.gridworld.world_size[0]**2+self.gridworld.world_size[1]**2))
        print("camera position",self.w.opts['center'])
        self.w.show()                                   # show the window
        self.w.setBackgroundColor('k')                  # set background color, option
        self.w.raise_()                                 # bring window to the front

        self.draw_grid()                                # draw the gridlines
        self.draw_boundaries()                          # draw boundaries of gridworld
        self.draw_axes()                                # draw grid axes
        self.draw_obstacles()                           # draw obstacles

        # draw agents and goals
        self.draw_agents()
        self.draw_goals()

    def draw_grid(self):
        """
        Desc: Visualize the grid

        Input(s):
            none
        Output(s):
            none
        """
        grid = gl.GLGridItem()
        grid.setSize(x=self.gridworld.world_size[0],y=self.gridworld.world_size[1])
        grid.setSpacing(x=self.gridworld.world_delta, y=self.gridworld.world_delta)
        grid.translate(self.gridworld.world_size[0]/2., self.gridworld.world_size[1]/2., 0.)    # translate so that it starts at (0,0)

        self.w.addItem(grid)

    def get_boundary(self):
        """
        Desc: creates the boundary mesh

        Input(s):
            none
        Output(s):
            bound_mesh: mesh of boundary
            bound_mesh_colors: boundary mesh colors
        """
        h = 1.*self.gridworld.world_delta # height of boundary
        d = 3.*self.gridworld.world_delta # thickness of boundary
        points = np.array([[0.0,0.0,0.0], # 0
                           [0.0,self.gridworld.world_size[1],0.0],
                           [self.gridworld.world_size[0],self.gridworld.world_size[1],0.0],
                           [self.gridworld.world_size[0],0.0,0.0],
                           [-d,-d,0.0],
                           [-d,self.gridworld.world_size[1]+d,0.0],
                           [self.gridworld.world_size[0]+d,self.gridworld.world_size[1]+d,0.0],
                           [self.gridworld.world_size[0]+d,-d,0.0],

                           [0.0,0.0,h], # 8
                           [0.0,self.gridworld.world_size[1],h],
                           [self.gridworld.world_size[0],self.gridworld.world_size[1],h],
                           [self.gridworld.world_size[0],0.0,h],
                           [-d,-d,h], # 12
                           [-d,self.gridworld.world_size[1]+d,h],
                           [self.gridworld.world_size[0]+d,self.gridworld.world_size[1]+d,h],
                           [self.gridworld.world_size[0]+d,-d,h]]) # 15
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
        bound_mesh_colors[:,:] = TRANSPARENT_GREY

        return bound_mesh,bound_mesh_colors

    def draw_boundaries(self):
        """
        Desc: Draw the gridworld boundaries

        Input(s):
            none
        Output(s):
            none
        """
        bound_mesh,bound_mesh_colors = self.get_boundary()      # create boundary mesh
        boundary = gl.GLMeshItem(vertexes=bound_mesh,           # defines the triangular mesh (Nx3x3)
                                 vertexColors=bound_mesh_colors,# defines mesh colors (Nx1)
                                 drawEdges=False,               # draw edges between mesh elements
                                 smooth=False)                  # speeds up rendering
                                 #computeNormals=False)         # speeds up rendering
        boundary.setGLOptions('additive')                       # allows them to be semi-transparent
        self.w.addItem(boundary)

    def draw_axes(self):
        """
        Desc: Draw the grid axes

        Input(s):
            none
        Output(s):
            none
        """
        # add all three colored axes
        xaxis_pts = np.array([[0.0,0.0,0.0],            # north axis start and end point
                        [1.1*self.gridworld.world_size[0],0.0,0.0]])
        xaxis = gl.GLLinePlotItem(pos=xaxis_pts,color=pg.glColor('r'),width=3.0)   # create line plot item
        self.w.addItem(xaxis)                           # add item to graph
        yaxis_pts = np.array([[0.0,0.0,0.0],            # east axis start and end point
                        [0.0,1.1*self.gridworld.world_size[1],0.0]])
        yaxis = gl.GLLinePlotItem(pos=yaxis_pts,color=pg.glColor('g'),width=3.0)   # create line plot item
        self.w.addItem(yaxis)                           # add item to graph
        zaxis_pts = np.array([[0.0,0.0,0.0],            # down axis start and end point
                        [0.0,0.0,5.0*self.gridworld.world_delta]])
        zaxis = gl.GLLinePlotItem(pos=zaxis_pts,color=pg.glColor('b'),width=3.0)   # create line plot item
        self.w.addItem(zaxis)

    def draw_obstacles(self):
        """
        Desc: draws the obstacles in the viewer

        Input(s):
            none
        Output(s):
            none
        """
        # draw map of the world: buildings
        self.fullMesh = np.array([], dtype=np.float32).reshape(0,3,3)
        self.fullMeshcolors = np.array([], dtype=np.float32).reshape(0,3,4)
        for obstacle in self.gridworld.obstacles:
            mesh, meshColors = self.generate_obstacle_mesh(obstacle.pos[0]+self.offset,
                                                     obstacle.pos[1]+self.offset)
            self.fullMesh = np.concatenate((self.fullMesh, mesh), axis=0)
            self.fullMeshcolors = np.concatenate((self.fullMeshcolors, meshColors), axis=0)
        self.obstacles_3d = gl.GLMeshItem(vertexes= self.fullMesh,  # defines the triangular mesh (Nx3x3)
                      vertexColors= self.fullMeshcolors,  # defines mesh colors (Nx1)
                      drawEdges=False,  # draw edges between mesh elements
                      smooth=False,  # speeds up rendering
                      computeNormals=False)  # speeds up rendering
        self.w.addItem(self.obstacles_3d)

    def generate_obstacle_mesh(self, x, y):
        """
        Desc: returns the mesh for an individual obstacle

        Input(s):
            x: x location of obstacle
            y: y location of obstacle
        Output(s):
            mesh:       mesh of obstacles
            meshColor:  color of mesh
        """
        width = self.gridworld.world_delta
        height = 1.*self.gridworld.world_delta*np.random.rand() + 0.5*self.gridworld.world_delta
        # define patches for an obstacle located at (x, y)
        # vertices of the obstacle
        points = np.array([[x + width / 2, y + width / 2, 0], #NE 0
                         [x + width / 2, y - width / 2, 0],   #SE 1
                         [x - width / 2, y - width / 2, 0],   #SW 2
                         [x - width / 2, y + width / 2, 0],   #NW 3
                         [x + width / 2, y + width / 2, height], #NE Higher 4
                         [x + width / 2, y - width / 2, height], #SE Higher 5
                         [x - width / 2, y - width / 2, height], #SW Higher 6
                         [x - width / 2, y + width / 2, height]]) #NW Higher 7
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

        meshColors = np.empty((10, 3, 4), dtype=np.float32)
        meshColors[:8] = GREY          # faces
        meshColors[8:] = LIGHT_GREY     # top
        return mesh, meshColors

    def draw_agents(self):
        """
        Desc: draws the agents in the viewer

        Input(s):
            none
        Output(s):
            none
        """
        self.agents_3d = []
        for agent in self.gridworld.agents:
            cyl_height = 2.*self.gridworld.world_delta
            cyl_radius = 0.8*(self.gridworld.world_delta/2.)
            cyl_object = gl.MeshData.cylinder(rows=100,cols=100,radius=[cyl_radius,cyl_radius],length=cyl_height)
            cyl_color = pg.glColor('r')
            agent_3d = gl.GLMeshItem(meshdata=cyl_object, smooth=True, drawFaces=True, drawEdges=False, color=cyl_color)
            agent_3d.translate(agent.pos[0]+self.offset,
                                agent.pos[1]+self.offset,
                                0)
            self.w.addItem(agent_3d)
            self.agents_3d.append(agent_3d)

    def draw_goals(self):
        """
        Desc: draws the goals in the viewer

        Input(s):
            none
        Output(s):
            none
        """
        self.goals_3d = []
        for goal in self.gridworld.goals:
            cyl_height = 2.*self.gridworld.world_delta
            cyl_radius = 0.8*(self.gridworld.world_delta/2.)
            cyl_object = gl.MeshData.cylinder(rows=100,cols=100,radius=[cyl_radius,cyl_radius],length=cyl_height)
            cyl_color = pg.glColor('g')
            goal_3d = gl.GLMeshItem(meshdata=cyl_object, smooth=True, drawFaces=True, drawEdges=False, color=cyl_color)
            goal_3d.translate(goal.pos[0]+self.offset,
                                goal.pos[1]+self.offset,
                                0)
            self.w.addItem(goal_3d)
            self.goals_3d.append(goal_3d)

    def run(self):
        """
        Desc: updates visualization

        Input(s):
            none
        Output(s):
            none
        """
        for i, agent in enumerate(self.gridworld.agents):
            if agent.next_action == 0:
                self.agents_3d[i].translate(-self.gridworld.world_delta,0,0)
            elif agent.next_action == 1:
                self.agents_3d[i].translate(self.gridworld.world_delta,0,0)
            elif agent.next_action == 2:
                self.agents_3d[i].translate(0,self.gridworld.world_delta,0)
            elif agent.next_action == 3:
                self.agents_3d[i].translate(0,-self.gridworld.world_delta,0)
            elif agent.next_action == None:
                pass
            agent.set_next_action(None)

        self.update_detected_obstacles()
        self.app.processEvents()

    def update_detected_obstacles(self):
        """
        Desc: updates the color of the immediate obstacles

        Input(s):
            none
        Output(s):
            none
        """
        immediate_obstacles_indexes = np.array([], dtype=int)
        for agent in self.gridworld.agents:
            immediate_obstacles_indexes = np.concatenate([immediate_obstacles_indexes, agent.obs])

        for obstacle_index in range(int(self.fullMeshcolors.shape[0]/10)):
            meshColors = np.empty((10, 3, 4), dtype=np.float32)
            if obstacle_index in immediate_obstacles_indexes:
                meshColors[:8] = YELLOW
                meshColors[8:] = DARK_YELLOW
            else:
                meshColors[:8] = GREY
                meshColors[8:] = LIGHT_GREY

            self.fullMeshcolors[obstacle_index*10:10*(obstacle_index+1),:,:] = meshColors

        self.obstacles_3d = gl.GLMeshItem(vertexes= self.fullMesh,  # defines the triangular mesh (Nx3x3)
                      vertexColors= self.fullMeshcolors,  # defines mesh colors (Nx1)
                      drawEdges=False,  # draw edges between mesh elements
                      smooth=False,  # speeds up rendering
                      computeNormals=False)  # speeds up rendering
