import numpy as np
import freud
import unittest
import util


class TestCluster(unittest.TestCase):

    def test_cluster_props(self):
        Nlattice = 4
        Nrep = 5

        positions = []
        for _ in range(Nrep):
            (box, pos) = util.make_fcc(Nlattice, Nlattice, Nlattice,
                                       noise=1e-2)
            positions.append(pos)

        # number of grid points (N = Nrep*Ngrid)
        Ngrid = positions[-1].shape[0]
        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster(box, 0.5)
        clust.computeClusters(positions, box=box)

        props = freud.cluster.ClusterProperties(box)
        props.computeProperties(positions, clust.cluster_idx, box=box)
        self.assertEqual(props.num_clusters, Ngrid)
        self.assertTrue(np.all(props.cluster_sizes == Nrep))

    def test_cluster_keys(self):
        Nlattice = 4
        Nrep = 5

        positions = []
        for _ in range(Nrep):
            (box, pos) = util.make_fcc(Nlattice, Nlattice, Nlattice,
                                       noise=1e-2)
            positions.append(pos)

        # number of grid points (N = Nrep*Ngrid)
        Ngrid = positions[-1].shape[0]
        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster(box, 0.5)
        clust.computeClusters(positions, box=box)
        clust.computeClusterMembership(np.array(range(Nrep*Ngrid)))

        self.assertEqual(len(clust.cluster_keys), Ngrid)

        ckeys = np.array(clust.cluster_keys) % Ngrid
        check_values = np.arange(Ngrid)[:, np.newaxis].repeat(Nrep, axis=1)

        self.assertTrue(np.all(ckeys == check_values))


if __name__ == '__main__':
    unittest.main()
