import matplotlib
import numpy as np
import numpy.testing as npt
import pytest
from numpy.lib import NumpyVersion
from StructureFactor_helper import (
    helper_partial_structure_factor_arguments,
    helper_test_accumulation,
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
    helper_test_S_0_is_N,
)

import freud

matplotlib.use("agg")


