import numpy.testing as npt
import numpy as np
import freud
import matplotlib
import unittest
import util
matplotlib.use('agg')


class TestHexatic(unittest.TestCase):
    def test_getK(self):
        hop = freud.order.Hexatic()
        npt.assert_equal(hop.k, 6)

    def test_getK_pass(self):
        k = 3
        hop = freud.order.Hexatic(k)
        npt.assert_equal(hop.k, 3)

    def test_order_size(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        hop.compute((box, points))
        npt.assert_equal(len(hop.particle_order), N)

    def test_compute_random(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        hop.compute((box, points))
        npt.assert_allclose(np.mean(hop.particle_order), 0. + 0.j, atol=1e-1)

    def test_compute(self):
        boxlen = 10
        r_max = 3
        box = freud.box.Box.square(boxlen)
        points = [[0.0, 0.0, 0.0]]

        for i in range(6):
            points.append([np.cos(float(i) * 2.0 * np.pi / 6.0),
                           np.sin(float(i) * 2.0 * np.pi / 6.0),
                           0.0])

        points = np.asarray(points, dtype=np.float32)
        points[:, 2] = 0.0
        hop = freud.order.Hexatic()

        # Test access
        hop.k
        with self.assertRaises(AttributeError):
            hop.particle_order

        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'nearest', r_max, 6, True)
        for nq, neighbors in test_set:
            hop.compute(nq, neighbors=neighbors)
            # Test access
            hop.k
            hop.particle_order

            npt.assert_allclose(hop.particle_order[0], 1. + 0.j, atol=1e-1)

    def test_weighted_random(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        voro = freud.locality.Voronoi()
        voro.compute(system=(box, points))

        # Ensure that \psi'_0 is 1
        hop = freud.order.Hexatic(k=0, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(hop.particle_order, 1., atol=1e-6)

        # Ensure that \psi'_1 is 0
        hop = freud.order.Hexatic(k=1, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 0., atol=1e-4)

        for k in range(0, 12):
            # Ensure that \psi'_k is between 0 and 1
            hop = freud.order.Hexatic(k=k, weighted=True)
            hop.compute(system=(box, points), neighbors=voro.nlist)
            order = np.absolute(hop.particle_order)
            assert (order >= 0).all() and (order <= 1).all()

            # Perform an explicit calculation in NumPy to verify results
            psi_k_weighted = np.zeros(len(points), dtype=np.complex)
            total_weights = np.zeros(len(points))
            rijs = box.wrap(points[voro.nlist.point_indices] -
                            points[voro.nlist.query_point_indices])
            thetas = np.arctan2(rijs[:, 1], rijs[:, 0])
            total_weights, _ = np.histogram(
                voro.nlist.query_point_indices,
                bins=len(points),
                weights=voro.nlist.weights)
            psi_k_weighted, _ = np.histogram(
                voro.nlist.query_point_indices,
                bins=len(points),
                weights=voro.nlist.weights * np.exp(thetas * k * 1.0j))
            psi_k_weighted /= total_weights

            npt.assert_allclose(hop.particle_order, psi_k_weighted, atol=1e-5)

    def test_weighted_square(self):
        unitcell = freud.data.UnitCell.square()
        box, points = unitcell.generate_system(num_replicas=10)
        voro = freud.locality.Voronoi()
        voro.compute(system=(box, points))

        # Ensure that \psi'_4 is 1
        hop = freud.order.Hexatic(k=4, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 1., atol=1e-5)

        # Ensure that \psi'_6 is 0
        hop = freud.order.Hexatic(k=6, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 0., atol=1e-5)

    def test_weighted_hex(self):
        unitcell = freud.data.UnitCell.hex()
        box, points = unitcell.generate_system(num_replicas=10)
        voro = freud.locality.Voronoi()
        voro.compute(system=(box, points))

        # Ensure that \psi'_6 is 1
        hop = freud.order.Hexatic(k=6, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 1., atol=1e-5)

        # Ensure that \psi'_4 is 0
        hop = freud.order.Hexatic(k=4, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 0., atol=1e-5)

    def test_3d_box(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=False)
        hop = freud.order.Hexatic()
        with self.assertRaises(ValueError):
            hop.compute((box, points))

    def test_repr(self):
        hop = freud.order.Hexatic(3)
        self.assertEqual(str(hop), str(eval(repr(hop))))

        hop = freud.order.Hexatic(7, weighted=True)
        self.assertEqual(str(hop), str(eval(repr(hop))))

    def test_repr_png(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        with self.assertRaises(AttributeError):
            hop.plot()
        self.assertEqual(hop._repr_png_(), None)

        hop.compute((box, points))
        hop._repr_png_()
        hop.plot()


if __name__ == '__main__':
    unittest.main()
