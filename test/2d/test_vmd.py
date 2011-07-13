from freud import trajectory

VMD.evaltcl('mol load hoomd start.xml dcd dump.dcd')
traj = trajectory.TrajectoryVMD()

for f in traj:
    print f.box.getLx(), f.box.getLy(), f.box.getLz(), f.box.is2D()
