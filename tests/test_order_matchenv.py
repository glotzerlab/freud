import numpy as np
import numpy.testing as npt
from freud.order import MatchEnv
from freud import trajectory
import unittest


class TestCluster(unittest.TestCase):
    #by Chrisy
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

    #by Chrisy
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


    #test MatchEnv.cluster function, defining clusters using constant k neighbors, hard_r=false, registration=false
    def test_cluster_kNeighbor(self):
        xyz = np.load("bcc.npy")
        xyz = np.array(xyz, dtype=np.float32)
        L = np.max(xyz)*2
        box = trajectory.Box(L, L, L, 0, 0, 0)

        rcut = 3.1
        kn = 14
        threshold = 0.1

        match = MatchEnv(box, rcut, kn)
        match.cluster(xyz, threshold, hard_r=False, registration=False)
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

    #test MatchEnv.cluster function, hard_r=true, registration=false
    def test_cluster_hardr(self):
        xyz = np.load("bcc.npy")
        xyz = np.array(xyz, dtype=np.float32)
        L = np.max(xyz)*2
        box = trajectory.Box(L, L, L, 0, 0, 0)

        rcut = 3.1
        kn = 14
        threshold = 0.1

        match = MatchEnv(box, rcut, kn)
        match.cluster(xyz, threshold, hard_r=True, registration=False)
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



    #test MatchEnv.cluster function, hard_r=false, registration=true
    def test_cluster_registration(self):
        xyz = np.load("bcc.N_1024.npy")
        xyz = np.array(xyz, dtype=np.float32)
        #define rotation matrix, rotate along z axis by pi/24 degree
        rotationAngle = np.pi/24.0
        R = np.array([[np.cos(rotationAngle), -np.sin(rotationAngle), 0], [np.sin(rotationAngle), np.cos(rotationAngle),0],[0,0,1 ]], float)
        #rotate particles that y>0, introduce grain boundary
        for i in range(len(xyz)):
            if xyz[i,1] < 0.0:
                xyz[i] = R.dot(xyz[i])


        L = np.max(xyz)*2
        box = trajectory.Box(L, L, L, 0, 0, 0)

        rcut = 3.1
        kn = 14
        threshold = 0.08

        match = MatchEnv(box, rcut, kn)
        match.cluster(xyz, threshold, hard_r=False, registration=True)
        clusters = match.getClusters()

        cluster_env = {}
        for cluster_ind in clusters:
            if cluster_ind not in cluster_env:
                cluster_env[cluster_ind] = np.copy(np.array(match.getEnvironment(cluster_ind)))


        #get environment for each particle
        tot_env = match.getTotEnvironment()

        #particle with index 1 and 5 has opposite y position, they should have same local environment
        npt.assert_equal(clusters[1], clusters[5], err_msg="two points do not have similar environment")

        #particle 1 and particle5's local environment should match
        returnResult = match.isSimilar(tot_env[1], tot_env[5], 0.1, registration=True)
        unittestObj = unittest.TestCase()
        unittestObj.assertNotEqual(len(returnResult[1]), 0, msg="two environments are not similar")


if __name__ == '__main__':
    unittest.main()

