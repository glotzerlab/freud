# ---- TrajectoryXML_test.py ----
from freud import trajectory

# read in a .xml file
traj = trajectory.TrajectoryXML(['start.xml'])

frame = traj.getCurrentFrame()

print frame.get('position')
print frame.get('velocity')
print frame.get('typename')
print frame.get('mass')
