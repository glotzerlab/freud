# ---- TrajectoryXML_test.py ----
from freud import trajectory

# read in a .xml file
traj = trajectory.TrajectoryXML(['start0.xml','start1.xml','start2.xml'])

for frame in traj:
    print frame.get('position')
    print frame.get('velocity')
    #print frame.get('typename')
    print frame.get('mass')
