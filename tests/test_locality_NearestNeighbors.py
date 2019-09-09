import numpy as np
import numpy.testing as npt
from freud import locality, box, parallel
import unittest
import itertools
from util import make_box_and_random_points
parallel.setNumThreads(1)


class TestNearestNeighbors(unittest.TestCase):
    def test_box_methods(self):
        L = 10  # Box Dimensions
        r_max = 3  # Cutoff radius
        N = 1  # number of particles
        num_neighbors = 6

        # Initialize cell list
        cl = locality.NearestNeighbors(r_max, num_neighbors)

        fbox, points = make_box_and_random_points(L, N)
        cl.compute([L, L, L], points, points)

        self.assertEqual(cl.box, fbox)
        npt.assert_array_equal(cl.r_sq_list,
                               [[-1, -1, -1, -1, -1, -1]])
        npt.assert_array_equal(cl.wrapped_vectors,
                               [[[-1, -1, -1],
                                 [-1, -1, -1],
                                 [-1, -1, -1],
                                 [-1, -1, -1],
                                 [-1, -1, -1],
                                 [-1, -1, -1]]])

    def test_neighbor_count(self):
        L = 10  # Box Dimensions
        r_max = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize cell list
        cl = locality.NearestNeighbors(r_max, num_neighbors)

        fbox, points = make_box_and_random_points(L, N)
        cl.compute(fbox, points, points)

        self.assertEqual(cl.num_neighbors, num_neighbors)
        self.assertEqual(len(cl.getNeighbors(0)), num_neighbors)

    def test_vals(self):
        L = 10  # Box Dimensions
        r_max = 2  # Cutoff radius
        num_neighbors = 1

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(r_max, num_neighbors, strict_cut=False)

        points = np.zeros(shape=(2, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.r_sq_list
        npt.assert_equal(neighbor_list[0, 0], 1)
        npt.assert_equal(neighbor_list[1, 0], 0)
        npt.assert_equal(rsq_list[0, 0], 9.0)
        npt.assert_equal(rsq_list[1, 0], 9.0)

    def test_two_ways(self):
        L = 10  # Box Dimensions
        r_max = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize cell list
        cl = locality.NearestNeighbors(r_max, num_neighbors)

        fbox, points = make_box_and_random_points(L, N)
        cl.compute(fbox, points, points)

        neighbor_list = cl.getNeighborList()
        rsq_list = cl.r_sq_list
        # pick particle at random
        pidx = np.random.randint(N)
        nidx = np.random.randint(num_neighbors)
        npt.assert_equal(neighbor_list[pidx, nidx],
                         cl.getNeighbors(pidx)[nidx])
        npt.assert_equal(rsq_list[pidx, nidx], cl.getRsq(pidx)[nidx])

    def test_strict_vals(self):
        L = 10  # Box Dimensions
        r_max = 2  # Cutoff radius
        num_neighbors = 1

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(r_max, num_neighbors, strict_cut=True)

        points = np.zeros(shape=(2, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.r_sq_list
        npt.assert_equal(neighbor_list[0, 0], cl.UINTMAX)
        npt.assert_equal(neighbor_list[1, 0], cl.UINTMAX)
        npt.assert_allclose(rsq_list[0, 0], -1.0, atol=1e-6)
        npt.assert_allclose(rsq_list[1, 0], -1.0, atol=1e-6)

    def test_strict_cutoff(self):
        L = 10  # Box Dimensions
        r_max = 2.01  # Cutoff radius
        N = 3  # number of particles
        num_neighbors = 2

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(r_max, num_neighbors, strict_cut=True)

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]

        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.r_sq_list
        npt.assert_equal(neighbor_list[0, 0], 1)
        npt.assert_equal(neighbor_list[0, 1], cl.UINTMAX)
        npt.assert_allclose(rsq_list[0, 0], 1.0, atol=1e-6)
        npt.assert_allclose(rsq_list[0, 1], -1.0, atol=1e-6)

    def test_cheap_hexatic(self):
        """Construct a poor man's hexatic order parameter using
        NeighborList properties"""
        fbox = box.Box.square(10)

        # make a square grid
        xs = np.linspace(-fbox.Lx/2, fbox.Lx/2, 10, endpoint=False)
        positions = np.zeros((len(xs)**2, 3), dtype=np.float32)
        positions[:, :2] = np.array(list(itertools.product(xs, xs)),
                                    dtype=np.float32)

        nn = locality.NearestNeighbors(1.5, 4)
        nn.compute(fbox, positions, positions)

        rijs = positions[nn.nlist.point_index] - \
            positions[nn.nlist.query_point_index]
        fbox.wrap(rijs)
        thetas = np.arctan2(rijs[:, 1], rijs[:, 0])

        cplx = np.exp(4*1j*thetas)
        psi4 = np.add.reduceat(cplx,
                               nn.nlist.segments) / nn.nlist.neighbor_counts

        self.assertEqual(len(psi4), len(positions))
        self.assertTrue(np.allclose(np.abs(psi4), 1))

    def test_repeated_neighbors(self):
        L = 10  # Box Dimensions
        N = 40  # number of particles
        num_neighbors = 6

        fbox, pos = make_box_and_random_points(L, N)

        for r_max in np.random.uniform(L/2, L/5, 128):
            # Initialize cell list
            cl = locality.NearestNeighbors(r_max, num_neighbors)
            nlist = cl.compute(fbox, pos, pos).nlist

            all_pairs = list(set(zip(nlist.query_point_index,
                                     nlist.point_index)))

            if len(all_pairs) != len(nlist):
                raise AssertionError(
                    'Repeated neighbor pair sizes in test_repeated_neighbors, '
                    'r_max={}'.format(r_max))

    def test_small_box(self):
        L = 10  # Box Dimensions
        N = 8  # number of particles
        num_neighbors = N - 1

        fbox, pos = make_box_and_random_points(L, N)

        for box_cell_count in range(2, 8):
            r_max = L/box_cell_count/1.0001
            # Initialize cell list
            cl = locality.NearestNeighbors(r_max, num_neighbors)
            nlist = cl.compute(fbox, pos, pos).nlist

            # all particles should be all particles' neighbors
            if len(nlist) != N*(N - 1):
                raise AssertionError(
                    'Wrong-sized neighbor list in test_even_cells,'
                    'box_cell_count={}'.format(box_cell_count))

    def test_single_neighbor(self):
        pos = np.zeros((10, 3), dtype=np.float32)
        pos[::2, 0] = 2*np.arange(5)
        pos[1::2, 0] = pos[::2, 0] + .25
        pos[:, 1] = pos[:, 0]
        pos[0] = (9, 7, 0)

        fbox = box.Box.square(L=4*len(pos))
        nn = locality.NearestNeighbors(1, 1).compute(fbox, pos, pos)
        nlist = nn.nlist

        self.assertEqual(len(nlist), len(pos))


if __name__ == '__main__':
    unittest.main()
