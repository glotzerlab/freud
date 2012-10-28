from freud import trajectory
from freud import interface
import random 
import numpy
import time

N=32000
L=55.0

list1 = numpy.zeros(shape=(N,3), dtype=numpy.float32)
list2 = numpy.zeros(shape=(N,3), dtype=numpy.float32)

box = trajectory.Box(L)

for i in xrange(0,N):
    list1[i,0] = (random.random() - 0.5) * L
    list1[i,1] = (random.random() - 0.5) * L
    list1[i,2] = (random.random() - 0.5) * L
    list2[i,0] = (random.random() - 0.5) * L
    list2[i,1] = (random.random() - 0.5) * L
    list2[i,2] = (random.random() - 0.5) * L

# Benchmark InterfaceMeasure
trials = 5
avg_time = 0

# Warm up
im = interface.InterfaceMeasure(box, 1)
im.compute(list1, list2)

start = time.time()
for trial in xrange(0,trials):
    print 'result = ' + str(im.compute(list1, list2))
    print 'result = ' + str(im.compute(list2, list1))
end = time.time()
print 'avg time per trial:', (end-start)/float(trials)
