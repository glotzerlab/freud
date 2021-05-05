import os
import sys

import numpy as np
import numpy.testing as npt
import pytest

import freud

try:
    import MDAnalysis

    MDANALYSIS = True
except ImportError:
    MDANALYSIS = False


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
    @pytest.mark.skipif(not MDANALYSIS, reason="MDAnalysis is not installed.")
    def test_mdanalysis_gsd(self):
        reader = MDAnalysis.coordinates.GSD.GSDReader(LJ_GSD)
        self.run_analyses(reader)

    @pytest.mark.skipif(not MDANALYSIS, reason="MDAnalysis is not installed.")
    def test_mdanalysis_dcd(self):
        reader = MDAnalysis.coordinates.DCD.DCDReader(LJ_DCD)
        self.run_analyses(reader)

    def test_gsd_gsd(self):
        import gsd.hoomd

        with gsd.hoomd.open(LJ_GSD, "rb") as traj:
            self.run_analyses(traj)

    def test_garnett_gsd(self):
        import garnett

        with garnett.read(LJ_GSD) as traj:
            self.run_analyses(traj)

    @pytest.mark.filterwarnings("ignore:Failed to import dcdreader")
    def test_garnett_dcd(self):
        import garnett

        with garnett.read(LJ_DCD) as traj:
            self.run_analyses(traj)
