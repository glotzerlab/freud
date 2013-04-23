from freud.trajectory import TrajectoryXML

import numpy as np
import numpy.testing as npt
import unittest

class TestTrajectoryXML(unittest.TestCase):
    def setUp(self):
        self.filenames = ["./data_trajectory_TrajectoryXML/full_output_1.xml",
                          "./data_trajectory_TrajectoryXML/full_output_2.xml"]
        
        self.expected = {}
        self.expected['position'] = np.asarray([[-1.45, 2.21, 1.56], [8.76, 1.02, 5.60], [5.67, 8.30, 4.67]])
        self.expected['velocity'] = np.asarray([[-0.5, -1.2, 0.4], [0.6, 2.0, 0.01], [-0.4, 3.0, 0.0]])        
        self.expected['image'] = np.asarray([[-1, -5, 12], [18, 2, -10], [13, -5, 0]])
        self.expected['acceleration'] = np.asarray([[-1.5, -1.2, 1.4], [0.6, 2.5, 0.41], [-0.4, 3.5, 2.0]])
        self.expected['mass'] = np.asarray([1.0, 2.0, 5.2])
        self.expected['diameter'] = np.asarray([3.2, 2.0, 1.5])
        self.expected['charge'] = np.asarray([1.0, 2.3, -3.0])
        self.expected['body'] = np.asarray([-1, 0, 0])
        self.expected['orientation'] = np.asarray([[1, 0, 0, 0], [0.0333, -0.0667, 0.1000, 0.1333], [0.5, 0.3, -1.9, 1.0]])
        self.expected['moment_inertia'] = np.asarray([[1, 0, 0, 2, 0, 3], [1, 2, 3, 4, 5, 6], [2, 3, 4, 1, 3, 2]])
        self.expected['typename'] = ["A", "long_type_name", "A"]
        self.expected['typeid'] = [0, 1, 0]
        
    
    def test_single_file(self):
        traj = TrajectoryXML(self.filenames[0:1])
        frame = traj[0]
        supported_props = traj.supported_props
        
        for prop in supported_props:
            if prop in self.expected:
                npt.assert_almost_equal(frame.get(prop), self.expected[prop], decimal=2, err_msg=prop)
                
        # Type special case
        self.assertEqual(frame.get('typename'), self.expected['typename'], msg="type name")
        self.assertEqual(frame.get('typeid'), self.expected['typeid'], msg="type id")
        
    def test_multiple_file_not_dynamic(self):
        
        traj = TrajectoryXML(self.filenames)
        frame = traj[1]
        supported_props = traj.supported_props
        
        for prop in supported_props:
            if prop == 'position':
                continue
            if prop in self.expected:
                npt.assert_almost_equal(frame.get(prop), self.expected[prop], decimal=2, err_msg=prop)
                
        self.assertEqual(frame.get('typename'), self.expected['typename'], msg="type name")
        self.assertEqual(frame.get('typeid'), self.expected['typeid'], msg="type id")
        
    def test_multiple_file_dynamic(self):
        
        traj = TrajectoryXML(self.filenames, dynamic = ['position', 'velocity', 'image', 'acceleration', 
                                                   'mass', 'diameter', 'charge', 'body', 
                                                   'orientation', 'moment_inertia', 'type'])
        frame = traj[1]
        supported_props = traj.supported_props
        
        expected = {}
        expected['position'] = np.asarray([[-2.45, 2.21, 1.56], [8.76, 1.12, 5.60], [5.67, 8.30, 4.77]])
        expected['velocity'] = np.asarray([[-10.5, -1.2, 0.4], [0.6, 12.0, 0.01], [-0.4, 3.0, 10.0]])
        expected['image'] = np.asarray([[-12, -5, 12], [18, 22, -10], [13, -5, 02]])
        expected['acceleration'] = np.asarray([[-1.25, -1.2, 1.4], [0.6, 2.25, 0.41], [-0.4, 3.5, 2.20]])
        expected['mass'] = np.asarray([11.0, 2.0, 5.2])
        expected['diameter'] = np.asarray([13.2, 2.0, 1.5])
        expected['charge'] = np.asarray([11.0, 2.3, -3.0])
        expected['body'] = np.asarray([-11, 2, 0])
        expected['orientation'] = np.asarray([[1, 2, 0, 0], [0.0333, -0.5667, 0.1000, 0.1333], [0.5, 0.3, -1.9, 1.1]])
        expected['moment_inertia'] = np.asarray([[1, 2, 0, 2, 0, 3], [1, 2, 4, 4, 5, 6], [2, 3, 4, 5, 3, 2]])
        
        for prop in supported_props:
            if prop in expected:
                npt.assert_almost_equal(frame.get(prop), expected[prop], decimal=2, err_msg=prop)
                
        # Type special case
        expected['typename'] = ["B", "long_type_name", "A"]
        expected['typeid'] = [1, 2, 0]
        self.assertEqual(frame.get('typename'), expected['typename'], 
                         msg="type name: %s\n%s" % (frame.get('typename'), expected['typename']))
        self.assertEqual(frame.get('typeid'), expected['typeid'], 
                         msg="type name: %s\n%s" % (frame.get('typeid'), expected['typeid']))
        
    
    def test_exceptions(self):
        # When setting dynamic to unsupported properties
        self.assertRaises(KeyError, TrajectoryXML, [""], dynamic=["not_supported"])
        
        
if __name__ == '__main__':
    unittest.main()       
        