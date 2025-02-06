# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import freud


class TestParallel:
    """Ensure that setting threads is appropriately reflected in Python."""

    # The setup and teardown ensure that these tests don't affect others.
    def setup_method(self):
        freud.parallel.set_num_threads(0)

    def teardown_method(self):
        freud.parallel.set_num_threads(0)

    def test_set(self):
        """Test setting the number of threads."""
        assert freud.parallel.get_num_threads() == 0
        freud.parallel.set_num_threads(3)
        assert freud.parallel.get_num_threads() == 3

    def test_NumThreads(self):
        """Test the NumThreads context manager."""
        assert freud.parallel.get_num_threads() == 0

        freud.parallel.set_num_threads(1)
        assert freud.parallel.get_num_threads() == 1

        with freud.parallel.NumThreads(2):
            assert freud.parallel.get_num_threads() == 2

        # After the context manager, the number of threads should revert
        # to its previous value.
        assert freud.parallel.get_num_threads() == 1
