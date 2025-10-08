# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import freud


def nlist_lifetime_check(get_nlist_func):
    """Ensure nlist exists past the lifetime of the compute that created it."""
    L = 10
    N = 100
    sys = freud.data.make_random_system(L, N, seed=1)

    nlist = get_nlist_func(sys)

    assert nlist.point_indices is not None
    assert nlist.query_point_indices is not None
    assert nlist.neighbor_counts is not None
