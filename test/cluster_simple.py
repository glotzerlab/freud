from freud import locality
from freud import trajectory
import numpy

# place particles 0, 1, and 4 next to each other
# place particles 2 and 5 next to each other
# place particle 3 all by itself
points = numpy.array([[0, 0, 0],
                      [0.5, 0.5, 0.5],
                      [10, 11, 12],
                      [-19, -12, -13],
                      [-0.5, 0.5, 0.5],
                      [10.5, 11.2, 11.3]], dtype='float32')


box = trajectory.Box(25);
cluster = locality.Cluster(box, 1.2);
cluster.computeClusters(points);

keys = numpy.array([0, 1, 0, 2, 3, 3], dtype=numpy.uint32);
cluster.computeClusterMembership(keys);

print "Num clusters:", cluster.getNumClusters();

print "cluster_idx:", cluster.getClusterIdx();

print "cluster_keys:", cluster.getClusterKeys();
