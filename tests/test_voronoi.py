from freud import box, voronoi
import numpy as np
import numpy.testing as npt
import unittest

class TestVoronoi(unittest.TestCase):
    def test_basic(self):
        # Test that voronoi tesselations of random systems have the same
        # number of points and polytopes
        L = 10 # Box length
        N = 50 # Number of particles
        fbox = box.Box.square(L) # Initialize box
        vor = voronoi.Voronoi(fbox)
        positions = np.random.uniform(-L/2, L/2, size=(N, 2)) # Generate random points in the box
        positions = np.insert(positions, 2, 0, axis=1).astype(np.float32) # Add a z-component of 0
        vor.compute(positions, buff=L/2)

        result = vor.getVoronoiPolytopes()

        npt.assert_equal(len(result), len(positions))

    def test_voronoi_tess_2d(self):
        # Test that the voronoi polytope works for a 2D system
        L = 10 # Box length
        fbox = box.Box.square(L)
        vor = voronoi.Voronoi(fbox)
        # Make a regular grid
        positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0],
                              [1, 0, 0], [1, 1, 0], [1, 2, 0],
                              [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.compute(positions)
        npt.assert_equal(vor.getVoronoiPolytopes(),
                [np.array([[ 1.5,  1.5, 0], [ 0.5,  1.5, 0],
                           [ 0.5,  0.5, 0], [ 1.5,  0.5, 0]])])

    def test_voronoi_tess_3d(self):
        # Test that the voronoi polytope works for a 3D system
        L = 10 # Box length
        fbox = box.Box.cube(L)
        vor = voronoi.Voronoi(fbox)
        # Make a regular grid
        positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0],
                              [1, 0, 0], [1, 1, 0], [1, 2, 0],
                              [2, 0, 0], [2, 1, 0], [2, 2, 0],
                              [0, 0, 1], [0, 1, 1], [0, 2, 1],
                              [1, 0, 1], [1, 1, 1], [1, 2, 1],
                              [2, 0, 1], [2, 1, 1], [2, 2, 1],
                              [0, 0, 2], [0, 1, 2], [0, 2, 2],
                              [1, 0, 2], [1, 1, 2], [1, 2, 2],
                              [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float32)
        vor.compute(positions)
        npt.assert_equal(vor.getVoronoiPolytopes(),
                [np.array([[1.5, 1.5, 1.5], [1.5, 0.5, 1.5], [1.5, 0.5, 0.5],
                           [1.5, 1.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 1.5],
                           [0.5, 1.5, 0.5], [0.5, 1.5, 1.5]])])

    def test_voronoi_neighbors(self):
        # Test that voronoi neighbors in the first and second shells are
        # correct in 2D
        L = 10 # Box length
        fbox = box.Box.square(L)
        vor = voronoi.Voronoi(fbox)
        # Make a regular grid
        positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0],
                              [1, 0, 0], [1, 1, 0], [1, 2, 0],
                              [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.computeNeighbors(positions)
        npt.assert_equal(vor.getNeighbors(1),
                [[1, 3], [0, 2, 4], [5, 1], [0, 6, 4], [3, 5, 1, 7], [8, 2, 4],
                 [3, 7], [6, 8, 4], [5, 7]])
        npt.assert_equal(vor.getNeighbors(2),
                [[1, 2, 3, 4, 6], [0, 2, 3, 4, 5, 7], [0, 1, 4, 5, 8],
                 [0, 1, 4, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8],
                 [1, 2, 3, 4, 7, 8], [0, 3, 4, 7, 8], [1, 3, 4, 5, 6, 8],
                 [2, 4, 5, 6, 7]])

    def test_nlist_symmetric(self):
        # Test that the voronoi neighborlist is symmetric
        L = 10 # Box length
        rbuf = 3 # Cutoff radius
        N = 40 # Number of particles

        fbox = box.Box.cube(L) # Initialize box
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        vor = voronoi.Voronoi(fbox)
        vor.computeNeighbors(points, fbox, rbuf)
        nlist = vor.getNeighborList()

        ijs = set(zip(nlist.index_i, nlist.index_j))
        jis = set(zip(nlist.index_j, nlist.index_i))

        # we shouldn't have duplicate (i, j) pairs in the
        # resulting neighbor list
        npt.assert_equal(len(ijs), len(nlist))
        npt.assert_equal(len(ijs), len(jis))

        # every (i, j) pair should have a corresponding (j, i) pair
        self.assertTrue(all((j, i) in jis for (i, j) in ijs))

if __name__ == '__main__':
    unittest.main()
