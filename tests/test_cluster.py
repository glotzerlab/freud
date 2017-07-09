import numpy as np
import numpy.testing as npt
import freud
import unittest
import internal

class TestCluster(unittest.TestCase):
    def test_basic(self):
        Nlattice = 4
        Nrep = 5

        positions = []
        for _ in range(Nrep):
            (box, pos) = internal.make_fcc(Nlattice, Nlattice, Nlattice, noise=1e-2)
            positions.append(pos)

        # number of grid points (N = Nrep*Ngrid)
        Ngrid = positions[-1].shape[0]
        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster(box, 0.5)
        clust.computeClusters(positions)

        props = freud.cluster.ClusterProperties(box)
        props.computeProperties(positions, clust.getClusterIdx())

        self.assertEqual(props.getNumClusters(), Ngrid)

        self.assertTrue(np.all(props.getClusterSizes() == Nrep))

if __name__ == '__main__':
    unittest.main()
