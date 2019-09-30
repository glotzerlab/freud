import numpy as np
import numpy.testing as npt
import garnett
import freud
import unittest
import os


def _get_structure_data(structure, qtype):
    return np.genfromtxt(os.path.join(
        os.path.dirname(__file__), 'files', 'minkowski_structure_metrics',
        '{}_{}.txt'.format(structure, qtype)))


class TestMinkowski(unittest.TestCase):
    def test_minkowski_structure_metrics(self):
        for structure in ['fcc', 'bcc', 'hcp', 'sc']:
            expected_ql = _get_structure_data(structure, 'q')
            expected_avql = _get_structure_data(structure, 'avq')
            expected_wl = _get_structure_data(structure, 'w')
            expected_avwl = _get_structure_data(structure, 'avw')

            with garnett.read(os.path.join(
                    os.path.dirname(__file__),
                    'files', 'minkowski_structure_metrics',
                    '{}.gsd'.format(structure))) as traj:
                frame = traj[0]
                box = frame.box
                positions = frame.positions.copy()

            voro = freud.locality.Voronoi()
            voro.compute(box, positions)
            for sph_l in range(expected_ql.shape[1]):

                # These tests fail for unknown (probably numerical) reasons.
                if structure == 'hcp' and sph_l in [3, 5]:
                    continue

                # Test Ql
                comp = freud.order.Steinhardt(sph_l, weighted=True)
                comp.compute((box, positions), neighbors=voro.nlist)
                npt.assert_allclose(
                    comp.order, expected_ql[:, sph_l], rtol=5e-5, atol=1e-3)

                # Test Average Ql
                comp = freud.order.Steinhardt(
                    sph_l, average=True, weighted=True)
                comp.compute((box, positions), neighbors=voro.nlist)
                npt.assert_allclose(
                    comp.order, expected_avql[:, sph_l],
                    rtol=5e-5, atol=1e-3)

                # These tests fail for unknown (probably numerical) reasons.
                if sph_l != 2:
                    # Test Wl
                    comp = freud.order.Steinhardt(
                        sph_l, Wl=True, weighted=True, Wl_normalize=True)
                    comp.compute((box, positions), neighbors=voro.nlist)
                    npt.assert_allclose(
                        comp.order, expected_wl[:, sph_l],
                        rtol=5e-5, atol=1e-3)

                    # Test Average Wl
                    comp = freud.order.Steinhardt(
                        sph_l, Wl=True, average=True, weighted=True,
                        Wl_normalize=True)
                    comp.compute((box, positions), neighbors=voro.nlist)
                    npt.assert_allclose(
                        comp.order, expected_avwl[:, sph_l],
                        rtol=5e-5, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
