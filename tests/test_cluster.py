import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import util
from test_managedarray import ManagedArrayTestBase

import freud

matplotlib.use("agg")


class TestCluster:
    def test_constructor(self):
        """Ensure no arguments are accepted to the constructor."""
        freud.cluster.Cluster()
        with pytest.raises(TypeError):
            freud.cluster.Cluster(42)

    def test_cluster_props(self):
        Nlattice = 4
        Nrep = 5

        np.random.seed(0)
        positions = []
        for _ in range(Nrep):
            (box, pos) = freud.data.UnitCell.fcc().generate_system(
                Nlattice, sigma_noise=1e-2
            )
            positions.append(pos)

        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster()
        # Test protected attribute access
        with pytest.raises(AttributeError):
            clust.num_clusters
        with pytest.raises(AttributeError):
            clust.num_particles
        with pytest.raises(AttributeError):
            clust.cluster_idx

        # Test with explicit box provided
        clust.compute((box, positions), neighbors={"r_max": 0.5})
        idx = np.copy(clust.cluster_idx)

        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "ball", 0.5, 0, True
        )
        for nq, neighbors in test_set:
            clust.compute(nq, neighbors=neighbors)
            assert np.all(clust.cluster_idx == idx)

        # Test if attributes are accessible now
        clust.num_clusters
        clust.cluster_idx

        # Test all property APIs
        props = freud.cluster.ClusterProperties()

        # Test protected attribute access
        with pytest.raises(AttributeError):
            props.num_clusters
        with pytest.raises(AttributeError):
            props.centers
        with pytest.raises(AttributeError):
            props.gyrations
        with pytest.raises(AttributeError):
            props.sizes

        props.compute((box, positions), clust.cluster_idx)

        # Test if attributes are accessible now
        props.centers
        props.gyrations
        props.sizes

        assert np.all(props.sizes == Nrep)

    def test_cluster_props_advanced(self):
        """Test radius of gyration and center of mass calculations"""
        box = freud.box.Box.square(L=5)
        positions = np.array([[0, -2, 0], [0, -2, 0], [0, 2, 0], [-0.1, 1.9, 0]])
        clust = freud.cluster.Cluster()
        clust.compute((box, positions), neighbors={"r_max": 0.5})

        props = freud.cluster.ClusterProperties()
        props.compute((box, positions), clust.cluster_idx)

        com_1 = [0, -2, 0]
        com_2 = [-0.05, 1.95, 0]
        g_tensor_2 = [[0.0025, 0.0025, 0], [0.0025, 0.0025, 0], [0, 0, 0]]
        rg_2 = np.sqrt(np.trace(g_tensor_2))
        npt.assert_allclose(props.centers[0, :], com_1, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(props.centers[1, :], com_2, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(props.gyrations[0], 0, atol=1e-5)
        npt.assert_allclose(props.gyrations[1], g_tensor_2, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(props.radii_of_gyration, [0, rg_2], rtol=1e-5, atol=1e-5)

    def test_cluster_different_masses(self):
        "Tests center of mass with varying point masses, instead of default of 1."
        box = freud.box.Box.square(L=5)
        positions = np.array([[0, -2, 0], [0, -2, 0], [0, 2, 0], [-0.1, 1.9, 0]])
        masses = [2, 1, 4, 1]

        clust = freud.cluster.Cluster()
        clust.compute((box, positions), neighbors={"r_max": 0.5})

        props = freud.cluster.ClusterProperties()
        props.compute((box, positions), clust.cluster_idx, masses)

    def test_cluster_com_periodic(self):
        "Tests center of mass for symmetric, box-spanning clusters."
        box = freud.Box.cube(3)

        # Center of mass is near the periodic boundary, not near the origin
        points = [
            [0.1, 0, 0],
            [-0.9, 0, 0],
            [-0.9, 0.5, 0],
            [-0.9, -0.5, 0],
            [1.1, 0, 0],
            [1.1, 0.5, 0],
            [1.1, -0.5, 0],
        ]

        clp = freud.cluster.ClusterProperties()
        clp.compute((box, points), np.zeros(len(points)))

        npt.assert_allclose(clp.centers, [[-1.4, 0, 0]], rtol=1e-5, atol=1e-5)

    def test_cluster_keys(self):
        Nlattice = 4
        Nrep = 5

        np.random.seed(0)
        positions = []
        for _ in range(Nrep):
            (box, pos) = freud.data.UnitCell.fcc().generate_system(
                Nlattice, sigma_noise=1e-2
            )
            positions.append(pos)

        # number of grid points (N = Nrep*Ngrid)
        Ngrid = positions[-1].shape[0]
        positions = np.array(positions).reshape((-1, 3))

        clust = freud.cluster.Cluster()

        # Test protected attribute access
        with pytest.raises(AttributeError):
            clust.cluster_keys

        clust.compute(
            (box, positions), keys=np.arange(Nrep * Ngrid), neighbors={"r_max": 0.5}
        )

        # Test if attributes are accessible now
        clust.cluster_keys

        assert len(clust.cluster_keys) == Ngrid

        ckeys = np.array(clust.cluster_keys) % Ngrid
        check_values = np.arange(Ngrid)[:, np.newaxis].repeat(Nrep, axis=1)

        assert np.all(ckeys == check_values)

    def test_repr(self):
        clust = freud.cluster.Cluster()
        assert str(clust) == str(eval(repr(clust)))
        props = freud.cluster.ClusterProperties()
        assert str(props) == str(eval(repr(props)))

    def test_repr_png(self):
        box = freud.box.Box.square(L=5)
        positions = np.array([[0, -2, 0], [0, -2, 0], [0, 2, 0], [-0.1, 1.9, 0]])
        clust = freud.cluster.Cluster()

        with pytest.raises(AttributeError):
            clust.plot()
        assert clust._repr_png_() is None

        clust.compute((box, positions), neighbors={"r_max": 0.5})
        clust._repr_png_()
        plt.close("all")


class TestClusterManagedArray(ManagedArrayTestBase):
    def build_object(self):
        self.obj = freud.cluster.Cluster()

    @property
    def computed_properties(self):
        return ["cluster_idx"]

    def compute(self):
        box = freud.box.Box.cube(10)
        num_points = 100
        points = np.random.rand(num_points, 3) * box.L - box.L / 2
        self.obj.compute((box, points), neighbors={"r_max": 2})
