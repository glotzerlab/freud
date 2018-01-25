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
from ._freud import NeighborList

class Voronoi:
    ## Compute the Voronoi tesselation of a 2D or 3D system using qhull
    # This essentially just wraps scipy.spatial.Voronoi, but accounts for
    # periodic boundary conditions

    def __init__(self, box, buff=0.1):
        ##Initialize Voronoi
        # \param box The simulation box
        self.box = box
        self.buff = buff

    def setBox(self, box):
        ##Set the simulation box
        self.box = box

    def setBufferWidth(self, buff):
        ##Set the box buffer width
        self.buff = buff

    def compute(self, positions, box=None, buff=None):
        ##Compute Voronoi tesselation
        # \param box The simulation box
        # \param buff The buffer of particles to be duplicated to simulated PBC, default=0.1

        # If box or buff is not specified, revert to object quantities
        if box is None:
            box = self.box
        if buff is None:
            buff = self.buff

        # Compute the buffer particles in c++
        vbuff = VoronoiBuffer(box)
        vbuff.compute(positions, buff)
        self.buff = buffer_parts = vbuff.getBufferParticles()
        if self.buff.size > 0:
            self.expanded_points = np.concatenate((positions, buffer_parts))
        else:
            self.expanded_points = positions

        # Use only the first two components if the box is 2D
        if box.is2D():
            self.expanded_points = self.expanded_points[:,:2]

        # Use qhull to get the points
        self.voronoi = qvoronoi(self.expanded_points)

        vertices = self.voronoi.vertices

        # Add a z-component of 0 if the box is 2D
        if box.is2D():
            vertices = np.insert(vertices, 2, 0, 1)

        # Construct a list of polygon/hedra vertices
        self.poly_verts = list()
        for region in self.voronoi.point_region[:len(positions)]:
            if -1 in self.voronoi.regions[region]:
                continue
            self.poly_verts.append(vertices[self.voronoi.regions[region]])
        return self;

    def getBuffer(self):
        # Return the list of voronoi polytope vertices
        return self.buff

    def getVoronoiPolytopes(self):
        # Return the list of voronoi polytope vertices
        return self.poly_verts

    def computeNeighbors(self, positions, box=None, buff=None):
        """Compute the neighbors of each particle based on the voronoi tessellation.
        One can include neighbors from multiple voronoi shells by specifying 'numShells' variable.
        An example code to compute neighbors up to two voronoi shells for a 2D mesh:

        vor = voronoi.Voronoi(box.Box(5, 5))
        pos = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        vor.computeNeighbors(pos)
        neighbors = vor.getNeighbors(2)

        Returns a list of lists of neighbors
        Note: input positions must be a 3D array. For 2D, set the z value to be 0.
        """

        # If box or buff is not specified, revert to object quantities
        if box is None:
            box = self.box
        if buff is None:
            buff = self.buff

        # Compute the buffer particles in c++
        vbuff = VoronoiBuffer(box)
        vbuff.compute(positions, buff)
        self.buff = buffer_parts = vbuff.getBufferParticles()

        if self.buff.size > 0:
            self.expanded_points = np.concatenate((positions, buffer_parts))
        else:
            self.expanded_points = positions

        if box.is2D():
            self.expanded_points = self.expanded_points[:,:2]

        # Use qhull to get the points
        self.voronoi = qvoronoi(self.expanded_points)
        ridge_points = self.voronoi.ridge_points
        ridge_vertices = self.voronoi.ridge_vertices
        vor_vertices = self.voronoi.vertices
        # Nearest neighbor index for each point
        self.firstShellNeighborList = [[]]*len(self.expanded_points)
        # Weight between nearest neighbors, which is the length of ridge between two points
        self.firstShellWeight = [[]]*len(self.expanded_points)
        for k in range(len(ridge_points)):
            # If -1 not in ridge_vertices, compute weight,
            # else do not include it in neighbor list.
            # -1 in ridge_vertices means the two particles are buffer particles.
            if ridge_vertices[k][0] != -1 and ridge_vertices[k][1] != -1:
                self.firstShellNeighborList[ridge_points[k,0]] = self.firstShellNeighborList[ridge_points[k,0]] + [ridge_points[k,1]]
                self.firstShellNeighborList[ridge_points[k,1]] = self.firstShellNeighborList[ridge_points[k,1]] + [ridge_points[k,0]]
                oneWeight = np.linalg.norm(vor_vertices[ridge_vertices[k][0]] - vor_vertices[ridge_vertices[k][1]])

                self.firstShellWeight[ridge_points[k,0]] = self.firstShellWeight[ridge_points[k,0]]+[oneWeight]
                self.firstShellWeight[ridge_points[k,1]] = self.firstShellWeight[ridge_points[k,1]]+[oneWeight]

    def getNeighbors(self, numShells):
        # Get numShells of neighbors for each particle
        neighbor_list = copy.copy(self.firstShellNeighborList)
        #delete [] in neighbor_list
        neighbor_list = [x for x in neighbor_list if len(x) > 0]
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

    def getNeighborList(self):
        # Build neighbor list based on voronoi neighbors
        neighbor_list = copy.copy(self.firstShellNeighborList)
        weight = copy.copy(self.firstShellWeight)

        # Count number of elements in neighbor_list
        count = 0
        for i in range(len(neighbor_list)):
            count += len(neighbor_list[i])

        # indexAry layout:
        # First column is reference particle index,
        # Second column is neighbor particle index,
        # Third column is weight = ridge length
        indexAry = np.zeros([count, 3], float)
        j = 0
        for i in range(len(neighbor_list)):
            N = len(neighbor_list[i])
            indexAry[j:j+N, 0] = i
            indexAry[j:j+N, 1] = np.array(neighbor_list[i])
            indexAry[j:j+N, 2] = np.array(weight[i])
            j += N

        result = NeighborList.from_arrays(
                len(neighbor_list), len(neighbor_list),
                indexAry[:,0], indexAry[:,1], weights=indexAry[:,2])
        return result
