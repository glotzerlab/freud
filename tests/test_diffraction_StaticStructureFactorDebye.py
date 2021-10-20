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
    def test_compute(self):
        sf = freud.diffraction.StaticStructureFactorDebye(1000, 100, 0)
        helper_test_compute(sf)

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

    def test_k_min(self):
        sf1 = freud.diffraction.StaticStructureFactorDebye(bins=100, k_max=10)
        sf2 = freud.diffraction.StaticStructureFactorDebye(bins=50, k_max=10, k_min=5)
        helper_test_k_min(sf1, sf2)
        with pytest.raises(ValueError):
            freud.diffraction.StaticStructureFactorDebye(bins=100, k_max=10, k_min=-1)

    def test_partial_structure_factor_arguments(self):
        sf = freud.diffraction.StaticStructureFactorDebye(1000, 100)
        helper_partial_structure_factor_arguments(sf)

    def test_partial_structure_factor_symmetry(self):
        """Compute a partial structure factor and ensure it is symmetric under
        type exchange."""
        sf = freud.diffraction.StaticStructureFactorDebye(bins=100, k_max=10, k_min=0)
        helper_test_partial_structure_factor_symmetry(sf)

    def test_partial_structure_factor_sum_normalization(self):
        """Ensure that the weighted sum of the partial structure factors is
        equal to the full scattering."""
        sf = freud.diffraction.StaticStructureFactorDebye(bins=100, k_max=10)
        helper_test_partial_structure_factor_sum_normalization(sf)

    def test_large_k_partial_cross_term_goes_to_zero(self):
        """Ensure S_{AB}(k) goes to zero at large k."""
        sf = freud.diffraction.StaticStructureFactorDebye(bins=5, k_max=1e6, k_min=1e5)
        helper_test_large_k_partial_cross_term_goes_to_zero(sf)

    def test_large_k_partial_self_term_goes_to_fraction(self):
        """Ensure S_{AA}(k) goes to N_A / N_total at large k."""
        sf = freud.diffraction.StaticStructureFactorDebye(bins=5, k_max=1e6, k_min=1e5)
        helper_test_large_k_partial_cross_term_goes_to_fraction(sf)

    def test_large_k_scattering_goes_to_one(self):
        """Ensure S(k) goes to one at large k."""
        sf = freud.diffraction.StaticStructureFactorDebye(bins=5, k_max=1e6, k_min=1e5)
        helper_test_large_k_partial_cross_term_goes_to_one(sf)

    def test_attribute_access(self):
        bins = 100
        k_max = 123
        k_min = 0.456
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        helper_test_attribute_access(sf, bins, k_max, k_min)

    @pytest.mark.skipif(
        NumpyVersion(np.__version__) < "1.15.0", reason="Requires numpy>=1.15.0."
    )
    def test_bin_precision(self):
        # Ensure bin edges and bounds are precise
        bins = 100
        k_max = 123
        k_min = 0.1
        sf = freud.diffraction.StaticStructureFactorDebye(
            bins=bins, k_max=k_max, k_min=k_min
        )
        helper_test_bin_precission(sf, bins, k_max, k_min)

    def test_min_valid_k(self):
        Lx = 10
        Ly = 8
        Lz = 7
        bins = 100
        k_max = 30
        k_min = 1
        min_valid_k = 4 * np.pi / Lz
        sf = freud.diffraction.StaticStructureFactorDebye(
            bins=bins, k_min=k_min, k_max=k_max
        )
        helper_test_min_valid_k(sf, min_valid_k, Lx, Ly, Lz)

    def test_attribute_shapes(self):
        bins = 100
        k_max = 123
        k_min = 0.456
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        helper_test_attribute_shapes(sf, bins, k_max, k_min)

    def test_repr(self):
        sf = freud.diffraction.StaticStructureFactorDebye(100, 123, 0.1)
        helper_test_repr(sf)
