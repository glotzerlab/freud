from freud import locality, box
import numpy as np
import numpy.testing as npt
import itertools
import unittest
import freud;freud.parallel.setNumThreads(1)

class TestNearestNeighbors(unittest.TestCase):
    def test_neighbor_count(self):
        L = 10 #Box Dimensions
        rcut = 3 #Cutoff radius
        N = 40; # number of particles
        num_neighbors = 6

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors)#Initialize cell list

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        cl.compute(fbox, points, points)

        self.assertEqual(cl.num_neighbors, num_neighbors)
        self.assertEqual(len(cl.getNeighbors(0)), num_neighbors)

    def test_vals(self):
        L = 10 #Box Dimensions
        rcut = 2 #Cutoff radius
        N = 2; # number of particles
        num_neighbors = 1

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors, strict_cut=False)#Initialize cell list

        points = np.zeros(shape=(2, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        npt.assert_equal(neighbor_list[0,0], 1)
        npt.assert_equal(neighbor_list[1,0], 0)
        npt.assert_equal(rsq_list[0,0], 9.0)
        npt.assert_equal(rsq_list[1,0], 9.0)

    def test_two_ways(self):
        L = 10 #Box Dimensions
        rcut = 3 #Cutoff radius
        N = 40; # number of particles
        num_neighbors = 6

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors)#Initialize cell list

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        cl.compute(fbox, points, points)

        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        # pick particle at random
        pidx = np.random.randint(N)
        nidx = np.random.randint(num_neighbors)
        npt.assert_equal(neighbor_list[pidx,nidx], cl.getNeighbors(pidx)[nidx])
        npt.assert_equal(rsq_list[pidx,nidx], cl.getRsq(pidx)[nidx])

    def test_strict_vals(self):
        L = 10 #Box Dimensions
        rcut = 2 #Cutoff radius
        N = 2; # number of particles
        num_neighbors = 1

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors, strict_cut=True)#Initialize cell list

        points = np.zeros(shape=(2, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        npt.assert_equal(neighbor_list[0,0], cl.UINTMAX)
        npt.assert_equal(neighbor_list[1,0], cl.UINTMAX)
        npt.assert_equal(rsq_list[0,0], -1.0)
        npt.assert_equal(rsq_list[1,0], -1.0)

    def test_strict_cutoff(self):
        L = 10 #Box Dimensions
        rcut = 2.01 #Cutoff radius
        N = 3; # number of particles
        num_neighbors = 2

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors, strict_cut=True)#Initialize cell list

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]

        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        npt.assert_equal(neighbor_list[0,0], 1)
        npt.assert_equal(neighbor_list[0,1], cl.UINTMAX)
        npt.assert_equal(rsq_list[0,0], 1.0)
        npt.assert_equal(rsq_list[0,1], -1.0)

    def test_cheap_hexatic(self):
        """Construct a poor man's hexatic order parameter using NeighborList properties"""
        box = freud.box.Box.square(10)

        # make a square grid
        xs = np.linspace(-box.getLx()/2, box.getLx()/2, 10, endpoint=False)
        positions = np.zeros((len(xs)**2, 3), dtype=np.float32)
        positions[:, :2] = np.array(list(itertools.product(xs, xs)), dtype=np.float32)

        nn = locality.NearestNeighbors(1.5, 4)
        nn.compute(box, positions, positions)

        rijs = positions[nn.nlist.index_j] - positions[nn.nlist.index_i]
        box.wrap(rijs)
        thetas = np.arctan2(rijs[:, 1], rijs[:, 0])

        cplx = np.exp(4*1j*thetas)
        psi4 = np.add.reduceat(cplx, nn.nlist.segments)/nn.nlist.neighbor_counts

        self.assertEqual(len(psi4), len(positions))
        self.assertTrue(np.allclose(np.abs(psi4), 1))

    def test_repeated_neighbors(self):
        L = 10 #Box Dimensions
        N = 40; # number of particles
        num_neighbors = 6

        fbox = box.Box.cube(L)#Initialize Box

        np.random.seed(0)

        pos = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)

        for rcut in np.random.uniform(L/2, L/5, 128):
            cl = locality.NearestNeighbors(rcut, num_neighbors)#Initialize cell list
            nlist = cl.compute(fbox, pos, pos).nlist

            all_pairs = list(set(zip(nlist.index_i, nlist.index_j)))

            if len(all_pairs) != len(nlist):
                raise AssertionError(
                    'Repeated neighbor pair sizes in test_repeated_neighbors, '
                    'rcut={}'.format(rcut))

    def test_small_box(self):
        L = 10 #Box Dimensions
        N = 8; # number of particles
        num_neighbors = N - 1

        fbox = box.Box.cube(L)#Initialize Box

        np.random.seed(0)

        pos = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)

        for box_cell_count in range(2, 8):
            rcut = L/box_cell_count/1.0001
            cl = locality.NearestNeighbors(rcut, num_neighbors)#Initialize cell list
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

        box = freud.box.Box.square(L=4*len(pos))
        nn = freud.locality.NearestNeighbors(1, 1).compute(box, pos, pos)
        nlist = nn.nlist

        self.assertEqual(len(nlist), len(pos))

if __name__ == '__main__':
    unittest.main()
