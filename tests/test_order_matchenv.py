import numpy as np
import numpy.testing as npt
from freud.order import MatchEnv
from freud import box
import unittest


class TestCluster(unittest.TestCase):
    #by Chrisy
    def test_single_cluster(self):
        xyz = np.load("bcc.npy")
        xyz = np.array(xyz, dtype=np.float32)
        L = np.max(xyz)*2
        fbox = box.Box.cube(L)

        rcut = 3.1
        kn = 14
        threshold = 0.1

        match = MatchEnv(fbox, rcut, kn)
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
        fbox = box.Box.cube(21)

        rcut = 4
        kn = 6
        threshold = 0.1

        match = MatchEnv(fbox, rcut, kn)
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
        fbox = box.Box.cube(L)

        rcut = 3.1
        kn = 14
        threshold = 0.1

        match = MatchEnv(fbox, rcut, kn)
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
        fbox = box.Box.cube(L)

        rcut = 3.1
        kn = 14
        threshold = 0.1

        match = MatchEnv(fbox, rcut, kn)
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
        box = box.Box(L, L, L, 0, 0, 0)

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

    #test MatchEnv.minimizeRMSD and registration functionality. overkill? maybe.
    def test_minimizeRMSD(self):
        env_vec = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,1],
                            [0,0,2]])
        threshold = 0.1

        # https://en.wikipedia.org/wiki/Rotation_matrix
        norm = np.array([1,1,1])
        norm = norm/np.sqrt(np.dot(norm, norm))

        theta = np.pi/10

        # r_cut and num_neigh are meaningless here
        r_cut = 2
        num_neigh = len(env_vec)

        ux = norm[0]
        uy = norm[1]
        uz = norm[2]
        c=np.cos(theta)
        s=np.sin(theta)

        R = np.array([[c+pow(ux, 2.)*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                      [uy*ux*(1-c)+uz*s, c+pow(uy, 2.)*(1-c), uy*uz*(1-c)-ux*s],
                      [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+pow(uz, 2.)*(1-c)]])

        ## 1. Grab the environment vectors.
        e0 = np.array(env_vec, dtype=np.single)
        # 1b. Set up a large enough box
        scale = 1.5
        rsq_arr = [np.dot(vec,vec) for vec in e0]
        rsq_max = max(rsq_arr)
        L = 2.*np.sqrt(rsq_max)*scale
        fbox = box.Box.cube(L)
        ## 2. Re-index the environment randomly to create a second environment.
        e1 = np.copy(e0)
        np.random.shuffle(e1)
        ## 3. Verify that OUR method isSimilar gives that these two environments are similar.
        match = MatchEnv(fbox, r_cut, num_neigh)
        [refPoints2, isSim_vec_map] = match.isSimilar(e0, e1, threshold, registration=False)
        npt.assert_almost_equal(e0, refPoints2[np.asarray(list(isSim_vec_map.values()))])
        ## 4. Calculate the minimal RMSD.
        [min_rmsd, refPoints2, minRMSD_vec_map] = match.minimizeRMSD(e0, e1, registration=False)
        ## 5. Verify that the minimizeRMSD method finds 0 minimal RMSD (with no registration.)
        npt.assert_equal(0.0, min_rmsd)
        ## 6. Verify that it gives the same vector mapping that isSimilar gave.
        npt.assert_equal(np.asarray(list(isSim_vec_map.values())), np.asarray(list(minRMSD_vec_map.values())))
        npt.assert_almost_equal(e0, refPoints2[np.asarray(list(minRMSD_vec_map.values()))])
        ## 7. Rotate the motif by a known rotation matrix. this matrix MUST be s.t. the minimal rmsd is the rmsd of the 1-1 mapping between the
        # vectors of the pre-rotated environment and the post-rotated environment (with no index-mixing).
        # This isn't guaranteed for any matrix, whatsoever. Work only with environments and rotations for which you know exactly what's going on here.
        # Be careful here : R rotates the position matrix where positions are COLUMN vectors
        e0_rot = np.array(np.transpose(np.dot(R, np.transpose(e0))), dtype=np.single)
        ## 8. Calculate the minimal RMSD assuming the 1-1 relationship.
        delta = e0_rot - e0
        deltasq = np.array([np.dot(vec,vec) for vec in delta])
        deltasum = np.sum(deltasq)
        deltasum /= len(deltasq)
        analy_rmsd = np.sqrt(deltasum)
        ## 9. Verify that minimizeRMSD gives this minimal RMSD (with no registration).
        [min_rmsd, refPoints2, minRMSD_vec_map] = match.minimizeRMSD(e0, e0_rot, registration=False)
        npt.assert_almost_equal(analy_rmsd, min_rmsd)
        npt.assert_almost_equal(e0_rot, refPoints2[np.asarray(list(minRMSD_vec_map.values()))])
        ## 10. Re-index the second environment randomly again.
        e1_rot = np.copy(e0_rot)
        np.random.shuffle(e1_rot)
        ## 11. Verify that minimizeRMSD gives this minimal RMSD again (with no registration).
        [min_rmsd, refPoints2, minRMSD_vec_map] = match.minimizeRMSD(e0, e1_rot, registration=False)
        npt.assert_almost_equal(analy_rmsd, min_rmsd)
        npt.assert_almost_equal(e0_rot, refPoints2[np.asarray(list(minRMSD_vec_map.values()))])
        ## 12. Now use minimizeRMSD with registration turned ON.
        [min_rmsd, refPoints2, minRMSD_vec_map] = match.minimizeRMSD(e0, e1_rot, registration=True)
        ## 13. This should get us back to 0 minimal rmsd.
        npt.assert_almost_equal(0., min_rmsd)
        npt.assert_almost_equal(e0, refPoints2[np.asarray(list(minRMSD_vec_map.values()))])
        ## 14. Finally use isSimilar with registration turned ON.
        [refPoints2, isSim_vec_map] = match.isSimilar(e0, e1_rot, threshold, registration=True)
        npt.assert_almost_equal(e0, refPoints2[np.asarray(list(isSim_vec_map.values()))])


if __name__ == '__main__':
    unittest.main()
