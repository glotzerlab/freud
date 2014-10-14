from freud import trajectory

# test that the XML reader properly identifies 2D boxes
print "XML only"
traj = trajectory.TrajectoryXMLDCD('start.xml', None)
f = traj[0];
print f.box.getLx(), f.box.getLy(), f.box.getLz(), f.box.is2D()

# test that the XML + DCD reader properly identifies 2D boxes
print
print "XML + DCD"
traj = trajectory.TrajectoryXMLDCD('start.xml', 'dump.dcd')
for f in traj:
    print f.box.getLx(), f.box.getLy(), f.box.getLz(), f.box.is2D()
