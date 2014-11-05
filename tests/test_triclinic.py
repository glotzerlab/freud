from freud import locality, trajectory
import numpy as np
import numpy.testing as npt
import unittest

class TestBox(unittest.TestCase):
    def test_BoxLength(self):
        box = trajectory.Box(2, 2, 2, 1, 0, 0)

        Lx = box.getLx()
        Ly = box.getLy()
        Lz = box.getLz()

        npt.assert_almost_equal(Lx, 2, decimal=2, err_msg="LxFail")
        npt.assert_almost_equal(Ly, 2, decimal=2, err_msg="LyFail")
        npt.assert_almost_equal(Lz, 2, decimal=2, err_msg="LzFail")

    def test_TiltFactor(self):
        box = trajectory.Box(2,	2, 2, 1, 0, 0);

        tiltxy = box.getTiltFactorXY()
        tiltxz = box.getTiltFactorXZ()
        tiltyz = box.getTiltFactorYZ()

        npt.assert_almost_equal(tiltxy, 1, decimal=2, err_msg="TiltXYfail")
        npt.assert_almost_equal(tiltxz, 0, decimal=2, err_msg="TiltXZfail")
        npt.assert_almost_equal(tiltyz, 0, decimal=2, err_msg="TiltYZfail")

    def test_BoxVolume(self):
        box = trajectory.Box(2, 2, 2, 1, 0, 0)

        volume = box.getVolume()

        npt.assert_almost_equal(volume, 8, decimal=2, err_msg="VolumnFail")

    def test_Wrap(self):
        box = trajectory.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[0, -1, -1], 
                               [0, 0.5, 0]], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0,0], -2, decimal=2, err_msg="WrapFail")

class TestXML(unittest.TestCase):
    def test_readBox2D(self):
        xmllist = ["triclinic2d.xml"]
        xml = trajectory.TrajectoryXML(xmllist)
        lx = xml.box.getLx()
        ly = xml.box.getLy()
        lz = xml.box.getLz()

        xy = xml.box.getTiltFactorXY()
        xz = xml.box.getTiltFactorXZ()
        yz = xml.box.getTiltFactorYZ()
        
        npt.assert_almost_equal(lx, 10.0, decimal=1, err_msg="XML2D_LxFail")
        npt.assert_almost_equal(ly, 10.0, decimal=1, err_msg="XML2D_LyFail")
        npt.assert_almost_equal(lz, 0.5, decimal=1, err_msg="XML2D_LzFail")
        npt.assert_almost_equal(xy, 0.09989, decimal=4, err_msg="XML_XYFail")
        npt.assert_almost_equal(xz, 0.0, decimal=1, err_msg="XML_XZFail")
        npt.assert_almost_equal(yz, 0.0, decimal=1, err_msg="XML_YZFail")

    def test_readBox3D(self):
        xmllist = ["triclinic.xml"]
        xml = trajectory.TrajectoryXML(xmllist)
        lx = xml.box.getLx()
        ly = xml.box.getLy()
        lz = xml.box.getLz()

        xy = xml.box.getTiltFactorXY()
        xz = xml.box.getTiltFactorXZ()
        yz = xml.box.getTiltFactorYZ()

        npt.assert_almost_equal(lx, 6.94586, decimal=4, err_msg="XML_LxFail")
        npt.assert_almost_equal(ly, 6.94586, decimal=4, err_msg="XML_LyFail")
        npt.assert_almost_equal(lz, 6.94586, decimal=4, err_msg="XML_LzFail")
        npt.assert_almost_equal(xy, 0.09989, decimal=4, err_msg="XML_XYFail")
        npt.assert_almost_equal(xz, 0.0, decimal=1, err_msg="XML_XZFail")
        npt.assert_almost_equal(yz, 0.0, decimal=1, err_msg="XML_YZFail")


class TestXMLDCD(unittest.TestCase):
    def test_readBoxFrame(self):
        xml = "triclinic.xml"
        dcd = "triclinic.dcd"
        traj = trajectory.TrajectoryXMLDCD(xml,dcd)

        lx = []
        ly = []
        lz = []
        xy = []
        xz = []
        yz = []

        expected_lx = [6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586]
        expected_ly = [6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586]
        expected_lz = [6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586]
        expected_xy = [0.0, 0.00989, 0.01989, 0.02989, 0.03989, 0.04990, 0.05990, 0.06990, 0.07990, 0.08989]
        expected_xz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected_yz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        for f in traj:
            lx.append(f.box.getLx())
            ly.append(f.box.getLy())
            lz.append(f.box.getLz())
            xy.append(f.box.getTiltFactorXY())
            xz.append(f.box.getTiltFactorXZ())
            yz.append(f.box.getTiltFactorYZ())

        npt.assert_almost_equal(lx, expected_lx, decimal=4, err_msg="XML_LxFail")
        npt.assert_almost_equal(ly, expected_ly, decimal=4, err_msg="XML_LyFail")
        npt.assert_almost_equal(lz, expected_lz, decimal=4, err_msg="XML_LzFail")
        npt.assert_almost_equal(xy, expected_xy, decimal=4, err_msg="XML_XYFail")
        npt.assert_almost_equal(xz, expected_xz, decimal=1, err_msg="XML_XZFail")
        npt.assert_almost_equal(yz, expected_yz, decimal=1, err_msg="XML_YZFail")

class TestPOS(unittest.TestCase):
    def test_readBoxFrame(self):
        pos = "triclinic.pos"
        traj = trajectory.TrajectoryPOS(pos)

        lx = []
        ly = []
        lz = []
        xy = []
        xz = []
        yz = []

        expected_lx = [6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586]
        expected_ly = [6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586]
        expected_lz = [6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586, 6.94586]
        expected_xy = [0.0, 0.00989, 0.01989, 0.02989, 0.03989, 0.04990, 0.05990, 0.06990, 0.07990, 0.08989]
        expected_xz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected_yz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        for f in traj:
            lx.append(f.box.getLx())
            ly.append(f.box.getLy())
            lz.append(f.box.getLz())
            xy.append(f.box.getTiltFactorXY())
            xz.append(f.box.getTiltFactorXZ())
            yz.append(f.box.getTiltFactorYZ())

        npt.assert_almost_equal(lx, expected_lx, decimal=4, err_msg="XML_LxFail")
        npt.assert_almost_equal(ly, expected_ly, decimal=4, err_msg="XML_LyFail")
        npt.assert_almost_equal(lz, expected_lz, decimal=4, err_msg="XML_LzFail")
        npt.assert_almost_equal(xy, expected_xy, decimal=4, err_msg="XML_XYFail")
        npt.assert_almost_equal(xz, expected_xz, decimal=1, err_msg="XML_XZFail")
        npt.assert_almost_equal(yz, expected_yz, decimal=1, err_msg="XML_YZFail")


if __name__ == '__main__':
    unittest.main()
