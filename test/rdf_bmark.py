from freud import trajectory
from freud import density
import random
import numpy
import time

N = 64000;
L = 55.0;

x = numpy.zeros(N, dtype='float32')
y = numpy.zeros(N, dtype='float32')
z = numpy.zeros(N, dtype='float32')

box = trajectory.Box(L);

for i in xrange(0,N):
    x[i] = (random.random() - 0.5) * L
    y[i] = (random.random() - 0.5) * L
    z[i] = (random.random() - 0.5) * L

# benchmark rdf
trials = 5;
avg_time = 0;

# warm up
rdf = density.RDF(box, 5.0, 0.05)
rdf.compute(x,y,z, x,y,z);

start = time.time();
for trial in xrange(0,trials):
    rdf.compute(x,y,z, x,y,z);
end = time.time();
print 'avg time per trial:', (end-start)/float(trials)
