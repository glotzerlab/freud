import numpy as np
import numpy.testing as npt
from freud.order import MatchEnv
from freud import trajectory
import unittest

class TestCluster(unittest.TestCase):
    def test_single_cluster(self):
        xyz = np.load("bcc.npy")
        xyz = np.array(xyz, dtype=np.float32)
        L = np.max(xyz)*2
        box = trajectory.Box(L, L, L, 0, 0, 0)

        rcut = 3.1
        kn = 14
        threshold = 0.1

        match = MatchEnv(box, rcut, kn)
        match.cluster(xyz, threshold)
        clusters = match.getClusters()

        cluster_env = {}
        for cluster_ind in clusters:
            if cluster_ind not in cluster_env:
                cluster_env[cluster_ind] = np.copy(np.array(match.getEnvironment(cluster_ind)))

        bcc_env = np.load("bcc_env.npy")
        num_cluster = len(cluster_env)
        env_cluster = cluster_env[0]

        npt.assert_equal(num_cluster, 1, err_msg="Number of cluster fail")
        npt.assert_almost_equal(env_cluster, bcc_env, decimal=2, err_msg="Cluster Environment fail")

    '''def test_double_cluster(self):
        xyz = np.load("fcc.npy")
        xyz = np.array(xyz, dtype=np.float32)
        Lx = np.max(xyz[:,0])*2
        Ly = np.max(xyz[:,1])*2
        Lz = np.max(xyz[:,2])*2
        box = trajectory.Box(Lx, Ly, Lz, 0, 0, 0)

        rcut = 3.1
        kn = 12
        threshold = 0.1

        '''

if __name__ == '__main__':
    unittest.main()

