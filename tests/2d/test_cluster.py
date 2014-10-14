from freud import trajectory
from freud import cluster
import numpy

# test that the cluster identification code can identify clusters in 2D
# this indirectly tests that LinkCell works properly in 2D as well

VMD.evaltcl('mol load hoomd start.xml dcd dump.dcd')
traj = trajectory.TrajectoryVMD()

f = traj[0]
cluster = cluster.Cluster(f.box, 2.0)

# get B particle index mask
type_A_mask = [(type=='A') for type in f.get('typename')];

for f in traj:
    pos = f.get('position');
    user = f.get('user');

    # select just B particles
    A_pos = numpy.compress(type_A_mask, pos, axis=0)

    cluster.computeClusters(A_pos);
    print "Num clusters:", cluster.getNumClusters();

    cluster_ids = cluster.getClusterIdx();
    numpy.place(user, type_A_mask, cluster_ids.astype(numpy.float32));
    user /= user.max();
    f.set('user', user);

