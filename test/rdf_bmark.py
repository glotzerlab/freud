from freud import trajectory
from freud import density
import random
import numpy
import time

N = 64000;
L = 55.0;

points = numpy.zeros(shape=(N,3), dtype=numpy.float32)

box = trajectory.Box(L);

for i in range(0,N):
    points[i,0] = (random.random() - 0.5) * L
    points[i,1] = (random.random() - 0.5) * L
    points[i,2] = (random.random() - 0.5) * L

# benchmark rdf
trials = 5;
avg_time = 0;

# warm up
rdf = density.RDF(box, 5.0, 0.05)
rdf.compute(points, points);

start = time.time();
for trial in range(0,trials):
    rdf.compute(points, points);
end = time.time();
print('avg time per trial: {}'.format((end-start)/float(trials)))
