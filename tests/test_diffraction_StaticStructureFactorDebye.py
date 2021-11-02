import matplotlib
import numpy as np
import numpy.testing as npt
import pytest
from numpy.lib import NumpyVersion
from StructureFactor_helper import (
    helper_partial_structure_factor_arguments,
    helper_test_attribute_access,
    helper_test_attribute_shapes,
    helper_test_bin_precission,
    helper_test_compute,
    helper_test_k_min,
    helper_test_large_k_partial_cross_term_goes_to_fraction,
    helper_test_large_k_partial_cross_term_goes_to_one,
    helper_test_large_k_partial_cross_term_goes_to_zero,
    helper_test_min_valid_k,
    helper_test_partial_structure_factor_sum_normalization,
    helper_test_partial_structure_factor_symmetry,
    helper_test_repr,
)

import freud

matplotlib.use("agg")


def _validate_debye_method(system, bins, k_max, k_min):
    """Validation of the static structure calculation.

    This method is a pure Python reference implementation of the Debye method
    implemented in C++ in freud.

    Args:
        system:
            Any object that is a valid argument to
            :class:`freud.locality.NeighborQuery.from_system`.
        bins (unsigned int):
            Number of bins in :math:`k` space.
        k_max (float):
            Maximum :math:`k` value to include in the calculation.
        k_min (float):
            Minimum :math:`k` value to include in the calculation.
    """
    system = freud.locality.NeighborQuery.from_system(system)
    N = len(system.points)

    Q = np.linspace(k_min, k_max, bins, endpoint=False)
    Q += (k_max - k_min) / bins / 2
    S = np.zeros_like(Q)

    # Compute all pairwise distances
    distances = system.box.compute_all_distances(system.points, system.points).flatten()

    for i, q in enumerate(Q):
        S[i] += np.sum(np.sinc(q * distances / np.pi)) / N

    return Q, S


class TestStaticStructureFactorDebye:
    def test_debye_validation(self):
        """Validate the Debye method against a Python implementation."""
        bins = 1000
        k_max = 100
        k_min = 0
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf.compute(system)
        Q, S = _validate_debye_method(system, bins, k_max, k_min)
        npt.assert_allclose(sf.bin_centers, Q)
        npt.assert_allclose(sf.S_k, S, rtol=1e-5, atol=1e-5)

    def test_debye_ase(self):
        """Validate Debye method agains ASE implementation"""
        ase = pytest.importorskip("ase")
        asexrd = pytest.importorskip("ase.utils.xrdebye")
        bins = 1000
        k_max = 100
        k_min = 0
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        # ase implementation has no PBC taken into account
        box.periodic = False
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        sf.compute(system)
        # ASE system generation as atoms object
        atoms = ase.Atoms(
            positions=points, pbc=True, cell=box.L, numbers=np.ones(len(points))
        )
        xrd = asexrd.XrDebye(
            atoms=atoms, wavelength=1.0, method=None, damping=0.0, alpha=1.0
        )
        # calculate S_k for given set of k values
        S_ase = xrd.calc_pattern(sf.bin_centers, mode="SAXS") / len(points)
        npt.assert_allclose(sf.S_k, S_ase, rtol=1e-5, atol=1e-5)

