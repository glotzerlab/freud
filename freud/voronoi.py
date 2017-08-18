# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

import numpy as np
import logging
logger = logging.getLogger(__name__)
import copy
try:
    from scipy.spatial import Voronoi as qvoronoi
except ImportError:
    qvoronoi = None
    msg = 'scipy.spatial.Voronoi is not available (requires scipy 0.12+), so freud.voronoi is not available.'
    logger.warning(msg)
    #raise ImportWarning(msg)
from ._freud import VoronoiBuffer

## Compute the Voronoi tesselation of a 2D or 3D system using qhull
# This essentially just wraps scipy.spatial.Voronoi, but accounts for
# periodic boundary conditions
class Voronoi:
    ##Initialize Voronoi
    # \param box The simulation box
    def __init__(self,box,buff=0.1):
        self.box=box
        self.buff=buff
    ##Set the simulation box
    def setBox(self,box):
        self.box=box
    ##Set the box buffer width
    def setBufferWidth(self,buff):
        self.buff=buff

    ##Compute Voronoi tesselation
    # \param box The simulation box
    # \param buff The buffer of particles to be duplicated to simulated PBC, default=0.1
    def compute(self,positions,box=None,buff=None):
        #if box or buff is not specified, revert to object quantities
        if box is None:
            box=self.box
        if buff is None:
            buff=self.buff

        #Compute the buffer particles in c++
        vbuff = VoronoiBuffer(box)
        vbuff.compute(positions,buff)
        self.buff = buffer_parts = vbuff.getBufferParticles()
        if self.buff != []:
            self.expanded_points = np.concatenate((positions,buffer_parts))
        else:
            self.expanded_points = positions

        #use qhull to get the points
        self.voronoi = qvoronoi(self.expanded_points)

        #construct a list of polygon/hedra vertices
        self.poly_verts=list()
        for region in self.voronoi.point_region[:len(positions)]:
                if -1 in self.voronoi.regions[region]:
                        continue
                self.poly_verts.append(self.voronoi.vertices[self.voronoi.regions[region]])
        return self;

    #return the list of voronoi polytope vertices
    def getBuffer(self):
        return self.buff

    #return the list of voronoi polytope vertices
    def getVoronoiPolytopes(self):
        return self.poly_verts

    """Compute the neighbors of each particle based on the voronoi tessalation.
    One can include neighbors from multiple voronoi shells by specifying 'numShells' variable.
    An example code to compute neighbors upto two voronoi shells for a 2D mesh

    vor = voronoi.Voronoi(box.Box(5, 5))
    pos = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
    vor.computeNeighbors(pos)
    neighbors = vor.getNeighbors(2)

    Returns a list of lists of neighbors
    """
    def computeNeighbors(self,positions,box=None,buff=None):
        #if box or buff is not specified, revert to object quantities
        if box is None:
            box=self.box
        if buff is None:
            buff=self.buff

        #Compute the buffer particles in c++
        vbuff = VoronoiBuffer(box)
        vbuff.compute(positions,buff)
        self.buff = buffer_parts = vbuff.getBufferParticles()

        if self.buff != []:
            self.expanded_points = np.concatenate((positions,buffer_parts))
        else:
            self.expanded_points = positions

        #use qhull to get the points
        self.voronoi = qvoronoi(self.expanded_points)
        ridge_points = self.voronoi.ridge_points
        self.firstShellNeighborList = [[]]*len(positions)
        for k in range(len(ridge_points)):
            self.firstShellNeighborList[ridge_points[k,0]] = self.firstShellNeighborList[ridge_points[k,0]] + [ridge_points[k,1]]
            self.firstShellNeighborList[ridge_points[k,1]] = self.firstShellNeighborList[ridge_points[k,1]] + [ridge_points[k,0]]

    #get numShells of neighbors for each particle
    def getNeighbors(self, numShells):
        neighbor_list = copy.copy(self.firstShellNeighborList)
        for _ in range(numShells-1):
            dummy_neighbor_list = copy.copy(neighbor_list)
            for i in range(len(neighbor_list)):
                numNeighbors = len(neighbor_list[i])
                for j in range(numNeighbors):
                    dummy_neighbor_list[i] = dummy_neighbor_list[i] + self.firstShellNeighborList[neighbor_list[i][j]]

                # remove duplicates
                dummy_neighbor_list[i] = list(set(dummy_neighbor_list[i]))
                try:
                    dummy_neighbor_list[i].remove(i)
                except:
                    pass
            neighbor_list = copy.copy(dummy_neighbor_list)

        return neighbor_list

        #construct a list of polygon/hedra vertices
        #self.poly_verts=list()
        #for region in self.voronoi.point_region[:len(positions)]:
        #        if -1 in self.voronoi.regions[region]:
        #                continue
        #        self.poly_verts.append(self.voronoi.vertices[self.voronoi.regions[region]])

    #return the list of voronoi polytope vertices
    #def getBuffer(self):
    #    return self.buff

    #return the list of voronoi polytope vertices
    #def getVoronoiPolytopes(self):
    #    return self.poly_verts
