from freud import trajectory
from freud import kspace
import numpy

# test that the RDF computation code can properly compute rdfs in 2D
traj = trajectory.TrajectoryXMLDCD('start.xml', 'dump.dcd')

f = traj[0]
kspace.SFactor3DPoints(f.box, 5);
