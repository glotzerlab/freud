# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import os
import sys

import gsd
import gsd.hoomd
import numpy as np
import numpy.testing as npt
import pytest

import freud

try:
    GSD_VERSION = gsd.__version__
    GSD_READ_FLAG = "rb"
except AttributeError:
    GSD_VERSION = gsd.version.version
    GSD_READ_FLAG = "r"


def _relative_path(*path):
    return os.path.join(os.path.dirname(__file__), *path)


LJ_GSD = _relative_path("files", "lj", "lj.gsd")
LJ_DCD = _relative_path("files", "lj", "lj.dcd")
LJ_RDF = np.load(_relative_path("files", "lj", "rdf.npz"))["rdf"]


class TestReaderIntegrations:
    def run_analyses(self, traj):
        """Run sample analyses that should work for any particle system."""
        rdf = freud.density.RDF(bins=100, r_max=5)
        ql = freud.order.Steinhardt(6)
        for system in traj:
            rdf.compute(system, reset=False)
            ql.compute(system, neighbors={"num_neighbors": 6})
        npt.assert_allclose(rdf.rdf, LJ_RDF, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        sys.platform.startswith("win"), reason="Not supported on Windows."
    )
    def test_mdanalysis_gsd(self):
        MDAnalysis = pytest.importorskip("MDAnalysis")
        reader = MDAnalysis.coordinates.GSD.GSDReader(LJ_GSD)
        self.run_analyses(reader)

    def test_mdanalysis_dcd(self):
        MDAnalysis = pytest.importorskip("MDAnalysis")
        reader = MDAnalysis.coordinates.DCD.DCDReader(LJ_DCD)
        self.run_analyses(reader)

    def test_gsd_gsd(self):
        with gsd.hoomd.open(LJ_GSD, GSD_READ_FLAG) as traj:
            self.run_analyses(traj)

    def test_ovito_gsd(self):
        import_file = pytest.importorskip("ovito.io").import_file
        pipeline = import_file(LJ_GSD)
        traj = [pipeline.compute(i) for i in range(pipeline.source.num_frames)]
        self.run_analyses(traj)
