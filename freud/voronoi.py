# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import numpy as np
import logging
import copy

from ._freud import VoronoiBuffer
from ._freud import NeighborList

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import Voronoi as qvoronoi
except ImportError:
    qvoronoi = None
    msg = ('scipy.spatial.Voronoi is not available (requires scipy 0.12+),'
           'so freud.voronoi is not available.')
    logger.warning(msg)


class Voronoi:
    # Compute the Voronoi tesselation of a 2D or 3D system using qhull
    # This essentially just wraps scipy.spatial.Voronoi, but accounts for
    # periodic boundary conditions

    def __init__(self, box, buff=0.1):
        # Initialize Voronoi
        # \param box The simulation box
        self.box = box
        self.buff = buff

    def setBox(self, box):
        # Set the simulation box
        self.box = box

    def setBufferWidth(self, buff):
        # Set the box buffer width
        self.buff = buff

    def compute(self, positions, box=None, buff=None):
        """ Compute Voronoi tesselation

        :param box: The simulation box
        :param buff: The buffer of particles to be duplicated to simulated
                     PBC, default=0.1
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

        # Use only the first two components if the box is 2D
        if box.is2D():
            self.expanded_points = self.expanded_points[:, :2]

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
        return self

    def getBuffer(self):
        # Return the list of voronoi polytope vertices
        return self.buff

    def getVoronoiPolytopes(self):
        # Return the list of voronoi polytope vertices
        return self.poly_verts

    def computeNeighbors(self, positions, box=None, buff=None):
        """Compute the neighbors of each particle based on the voronoi
        tessellation. One can include neighbors from multiple voronoi shells by
        specifying 'numShells' variable. An example code to compute neighbors
        up to two voronoi shells for a 2D mesh:

        Example::
            vor = voronoi.Voronoi(box.Box(5, 5))
            pos = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                [2, 0], [2, 1], [2, 2]])
            vor.computeNeighbors(pos)
            neighbors = vor.getNeighbors(2)

        Returns a list of lists of neighbors
        Note: input positions must be a 3D array. For 2D, set the z value to
            be 0.
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
            self.expanded_points = self.expanded_points[:, :2]

        # Use qhull to get the points
        self.voronoi = qvoronoi(self.expanded_points)
        ridge_points = self.voronoi.ridge_points
        ridge_vertices = self.voronoi.ridge_vertices
        vor_vertices = self.voronoi.vertices
        N = len(positions)

        # Nearest neighbor index for each point
        self.firstShellNeighborList = [[] for _ in range(N)]

        # Weight between nearest neighbors, which is the length of ridge
        # between two points in 2D or the area of the ridge facet in 3D
        self.firstShellWeight = [[] for _ in range(N)]
        for (k, (index_i, index_j)) in enumerate(ridge_points):
            if index_i >= N or index_j >= N:
                continue

            self.firstShellNeighborList[index_i].append(index_j)
            self.firstShellNeighborList[index_j].append(index_i)

            if -1 not in ridge_vertices[k]:
                if box.is2D():
                    # The weight for a 2D system is the
                    # length of the ridge line
                    weight = np.linalg.norm(
                        vor_vertices[ridge_vertices[k][0]] -
                        vor_vertices[ridge_vertices[k][1]])
                else:
                    # The weight for a 3D system is the ridge polygon area
                    # The process to compute this area is:
                    # 1. Project 3D polygon onto xy, yz, or zx plane,
                    #    by aligning with max component of the normal vector
                    # 2. Use shoelace formula to compute 2D area
                    # 3. Project back to get true area of 3D polygon
                    # See link below for sample code and further explanation
                    # http://geomalgorithms.com/a01-_area.html#area3D_Polygon()
                    vertex_coords = np.array([vor_vertices[i]
                                              for i in ridge_vertices[k]])

                    # Get a unit normal vector to the polygonal facet
                    r01 = vertex_coords[1] - vertex_coords[0]
                    r12 = vertex_coords[2] - vertex_coords[1]
                    norm_vec = np.cross(r01, r12)
                    norm_vec /= np.linalg.norm(norm_vec)

                    # Determine projection axis:
                    # c0 is the largest coordinate (x, y, or z) of the normal
                    # vector. We project along the c0 axis and use c1, c2 axes
                    # for computing the projected area.
                    c0 = np.argmax(np.abs(norm_vec))
                    c1 = (c0 + 1) % 3
                    c2 = (c0 + 2) % 3

                    vc1 = vertex_coords[:, c1]
                    vc2 = vertex_coords[:, c2]

                    # Use shoelace formula for the projected area
                    projected_area = 0.5*np.abs(
                        np.dot(vc1, np.roll(vc2, 1)) -
                        np.dot(vc2, np.roll(vc1, 1)))

                    # Project back to get the true area (which is the weight)
                    weight = projected_area / np.abs(norm_vec[c0])
            else:
                # This point was on the boundary, so as far as qhull
                # is concerned its ridge goes out to infinity
                weight = 0

            self.firstShellWeight[index_i].append(weight)
            self.firstShellWeight[index_j].append(weight)

    def getNeighbors(self, numShells):
        # Get numShells of neighbors for each particle
        neighbor_list = copy.copy(self.firstShellNeighborList)
        # delete [] in neighbor_list
        neighbor_list = [x for x in neighbor_list if len(x) > 0]
        for _ in range(numShells - 1):
            dummy_neighbor_list = copy.copy(neighbor_list)
            for i in range(len(neighbor_list)):
                numNeighbors = len(neighbor_list[i])
                for j in range(numNeighbors):
                    dummy_neighbor_list[i] = dummy_neighbor_list[i] + \
                        self.firstShellNeighborList[neighbor_list[i][j]]

                # remove duplicates
                dummy_neighbor_list[i] = list(set(dummy_neighbor_list[i]))

                if i in dummy_neighbor_list[i]:
                    dummy_neighbor_list[i].remove(i)

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
            indexAry[j:j + N, 0] = i
            indexAry[j:j + N, 1] = np.array(neighbor_list[i])
            indexAry[j:j + N, 2] = np.array(weight[i])
            j += N

        result = NeighborList.from_arrays(
            len(neighbor_list), len(neighbor_list),
            indexAry[:, 0], indexAry[:, 1], weights=indexAry[:, 2])
        return result
