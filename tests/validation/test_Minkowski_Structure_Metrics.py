import numpy as np
import pytest
from pytest import approx
import garnett
import freud


structures = ['fcc', 'bcc', 'hcp', 'sc']


@pytest.mark.parametrize("filename", structures)
def test_minkowski_structure_metric_ql(filename):
    expected_ql = np.genfromtxt('files/{}_q.txt'.format(filename))

    with garnett.read('files/{}.gsd'.format(filename)) as traj:
        frame = traj[0]
        box = frame.box
        positions = frame.positions
        voro = freud.locality.Voronoi()
        voro.compute(box, positions)
        for sph_l in range(expected_ql.shape[1]):
            comp = freud.order.Steinhardt(sph_l, weighted=True)
            comp.compute(box, positions, nlist=voro.nlist)
            assert comp.order == approx(expected_ql[:, sph_l],
                                        rel=5e-5, abs=1e-3)


if __name__ == '__main__':
    pytest.main()
