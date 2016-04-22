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

        # this is a nifty way of lexicographically sorting our arrays, for the purposes of an element-by-element comparison.
        # np.lexsort() sorts by the columns you feed it, with the final fed column being the "primary" sorting key.
        # getEnvironment() might return the motif with its vectors sorted any old way, and if we compare against a saved numpy
        # array then we have to order the two arrays in the same fashion.
        sorted_env_cluster = env_cluster[np.lexsort((env_cluster[:,0],env_cluster[:,1],env_cluster[:,2]))]
        sorted_bcc_env = bcc_env[np.lexsort((bcc_env[:,0],bcc_env[:,1],bcc_env[:,2]))]
        npt.assert_equal(num_cluster, 1, err_msg="Number of BCC cluster fail")
        npt.assert_almost_equal(sorted_env_cluster, sorted_bcc_env, decimal=5, err_msg="BCC Cluster Environment fail")

    def test_multi_cluster(self):
        xyz = np.load("sc.npy")
        xyz = np.array(xyz, dtype=np.float32)
        box = trajectory.Box(21, 21, 21, 0, 0, 0)

        rcut = 4
        kn = 6
        threshold = 0.1

        match = MatchEnv(box, rcut, kn)
        match.cluster(xyz, threshold)
        clusters = match.getClusters()

        cluster_env = {}
        for cluster_ind in clusters:
            if cluster_ind not in cluster_env:
                cluster_env[cluster_ind] = np.copy(np.array(match.getEnvironment(cluster_ind)))

        sc_env = np.load("sc_env.npy")
        # Randomly choose the 3rd cluster here to test
        env_cluster = cluster_env[2]
        num_cluster = len(cluster_env)

        # this is a nifty way of lexicographically sorting our arrays, for the purposes of an element-by-element comparison.
        # np.lexsort() sorts by the columns you feed it, with the final fed column being the "primary" sorting key.
        # getEnvironment() might return the motif with its vectors sorted any old way, and if we compare against a saved numpy
        # array then we have to order the two arrays in the same fashion.
        sorted_env_cluster = env_cluster[np.lexsort((env_cluster[:,0],env_cluster[:,1],env_cluster[:,2]))]
        sorted_sc_env = sc_env[np.lexsort((sc_env[:,0],sc_env[:,1],sc_env[:,2]))]
        npt.assert_equal(num_cluster, 6, err_msg="Number of SC cluster fail")
        npt.assert_almost_equal(sorted_env_cluster, sorted_sc_env, decimal=5, err_msg="SC Cluster Environment fail")


if __name__ == '__main__':
    unittest.main()

