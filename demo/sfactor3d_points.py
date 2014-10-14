from freud import trajectory
from freud import kspace
import numpy
import scipy.io

m = 5;
L = float(m + 1);

box = trajectory.Box(L,L,L);

points = numpy.zeros(shape=((m+1)*(m+1)*(m+1), 3), dtype=numpy.float32)
# setup a simple cubic array of points
c = 0;
for i in xrange(0,m+1):
    for j in xrange(0,m+1):
        for k in xrange(0,m+1):
            points[c,0] = float(i) - L/2.0;
            points[c,1] = float(j) - L/2.0;
            points[c,2] = float(k) - L/2.0;
            #print points[c]
            c += 1

g = 20;
sfac = kspace.SFactor3DPoints(box, g)

sfac.compute(points);

sfac_ana = kspace.AnalyzeSFactor3D(sfac.getS());
print sfac_ana.getPeakList(0.1);
peak_degen = sfac_ana.getPeakDegeneracy(0.1);
peaks = peak_degen.items();
peaks.sort();
for peak in peaks:
    print peak

S,q = sfac_ana.getSvsQ();
scipy.io.savemat('sfactor3d_points.mat', dict(S=S, q=numpy.array(q, dtype=numpy.float32)), oned_as='row');
