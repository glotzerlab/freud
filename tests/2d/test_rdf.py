from freud import trajectory
from freud import density
import numpy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import time

# test that the RDF computation code can properly compute rdfs in 2D
traj = trajectory.TrajectoryXMLDCD('start.xml', 'dump.dcd')

f = traj[0]
rdf = density.RDF(f.box, 20.0, 0.1)
total_rdf = rdf.getRDF();
n = 0;

start = time.time();
for f in traj:
    pos = f.get('position');

    rdf.compute(pos, pos);
    total_rdf += rdf.getRDF();
    n += 1;
    print n
end = time.time();
print 'avg time per frame:', (end-start)/float(n)

r = rdf.getR();
pyplot.plot(r, total_rdf/n)
pyplot.show();
