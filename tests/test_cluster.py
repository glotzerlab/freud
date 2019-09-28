import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestCluster(unittest.TestCase):
    def test_cluster_props(self):
        Nlattice = 4
        Nrep = 5

        np.random.seed(0)
        positions = []
        for _ in range(Nrep):
            (box, pos) = util.make_fcc(Nlattice, Nlattice, Nlattice,
                                       noise=1e-2)
            positions.append(pos)

        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster()
        # Test protected attribute access
        with self.assertRaises(AttributeError):
            clust.num_clusters
        with self.assertRaises(AttributeError):
            clust.num_particles
        with self.assertRaises(AttributeError):
            clust.cluster_idx

        # Test with explicit box provided
        clust.compute((box, positions), neighbors={'r_max': 0.5})
        idx = np.copy(clust.cluster_idx)

        test_set = util.make_raw_query_nlist_test_set_new(
            box, positions, positions, "ball", 0.5, 0, True)
        for nq, neighbors in test_set:
            clust.compute(nq, neighbors=neighbors)
            self.assertTrue(np.all(clust.cluster_idx == idx))

        # Test if attributes are accessible now
        clust.num_clusters
        clust.cluster_idx

        # Test all property APIs
        props = freud.cluster.ClusterProperties()

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            props.num_clusters
        with self.assertRaises(AttributeError):
            props.centers
        with self.assertRaises(AttributeError):
            props.gyrations
        with self.assertRaises(AttributeError):
            props.sizes

        props.compute(box, positions, clust.cluster_idx)

        # Test if attributes are accessible now
        props.centers
        props.gyrations
        props.sizes

        self.assertTrue(np.all(props.sizes == Nrep))

    def test_cluster_props_advanced(self):
        """Test radius of gyration and center of mass calculations"""
        box = freud.box.Box.square(L=5)
        positions = np.array([[0, -2, 0],
                              [0, -2, 0],
                              [0, 2, 0],
                              [-0.1, 1.9, 0]])
        clust = freud.cluster.Cluster()
        clust.compute((box, positions), neighbors={'r_max': 0.5})

        props = freud.cluster.ClusterProperties()
        props.compute(box, positions, clust.cluster_idx)

        com_1 = np.array([[0, -2, 0]])
        com_2 = np.array([[-0.05, 1.95, 0]])
        g_tensor_2 = np.array([[0.0025, 0.0025, 0],
                               [0.0025, 0.0025, 0],
                               [0, 0, 0]])
        self.assertTrue(np.all(props.centers[0, :] == com_1))
        self.assertTrue(np.allclose(props.centers[1, :], com_2))
        self.assertTrue(np.all(props.gyrations[0] == 0))
        self.assertTrue(np.allclose(props.gyrations[1], g_tensor_2))

    def test_cluster_keys(self):
        Nlattice = 4
        Nrep = 5

        np.random.seed(0)
        positions = []
        for _ in range(Nrep):
            (box, pos) = util.make_fcc(Nlattice, Nlattice, Nlattice,
                                       noise=1e-2)
            positions.append(pos)

        # number of grid points (N = Nrep*Ngrid)
        Ngrid = positions[-1].shape[0]
        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster()

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            clust.cluster_keys

        clust.compute((box, positions), keys=np.arange(Nrep*Ngrid),
                      neighbors={'r_max': 0.5})

        # Test if attributes are accessible now
        clust.cluster_keys

        self.assertEqual(len(clust.cluster_keys), Ngrid)

        ckeys = np.array(clust.cluster_keys) % Ngrid
        check_values = np.arange(Ngrid)[:, np.newaxis].repeat(Nrep, axis=1)

        self.assertTrue(np.all(ckeys == check_values))

    def test_repr(self):
        clust = freud.cluster.Cluster()
        self.assertEqual(str(clust), str(eval(repr(clust))))
        props = freud.cluster.ClusterProperties()
        self.assertEqual(str(props), str(eval(repr(props))))

    def test_repr_png(self):
        box = freud.box.Box.square(L=5)
        positions = np.array([[0, -2, 0],
                              [0, -2, 0],
                              [0, 2, 0],
                              [-0.1, 1.9, 0]])
        clust = freud.cluster.Cluster()

        with self.assertRaises(AttributeError):
            clust.plot()
        self.assertEqual(clust._repr_png_(), None)

        clust.compute((box, positions), neighbors={'r_max': 0.5})
        clust._repr_png_()

    def test_saved_values(self):
        """Check that saved output don't get overwritten by later calls to
        compute or object deletion."""
        L = 10
        num_points = 100
        num_tests = 3

        r_max = 2
        copied = []
        accessed = []
        cl = freud.cluster.Cluster()

        box = freud.box.Box.cube(L)
        for i in range(num_tests):
            points = np.random.rand(num_points, 3)*L - L/2
            cl.compute((box, points), neighbors={'r_max': r_max})

            copied.append(np.copy(cl.cluster_idx))
            accessed.append(cl.cluster_idx)
        npt.assert_array_equal(copied, accessed)

        del cl
        npt.assert_array_equal(copied, accessed)


if __name__ == '__main__':
    unittest.main()
