import numpy as np
import numpy.testing as npt
from freud import locality, box
from collections import Counter
import itertools
import sys
import unittest
import warnings


class TestSpatialDataAABB(unittest.TestCase):
    def test_query_ball(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        aq = locality.AABBQuery(fbox, points)
        print(aq.query_ball(points[[2]], 1.01))
        print(aq.query_ball(points, 1.01))
        print(aq.query(points[[1]], 3, 0.5, 1.1))


class TestSpatialDataLinkCell(unittest.TestCase):
    def test_query_ball(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        lc = locality.LinkCell(fbox, rcut, points)
        print(lc.query_ball(points[[2]], 1.01))
        print(lc.query_ball(points, 1.01))
        print(lc.query(points[[1]], 3))
