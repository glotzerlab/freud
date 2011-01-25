from freud import locality
from freud import trajectory
import numpy

# place particles 0 and 1 in cell 0
# place particle 2 in cell 1
# place particles 3,4,5 in cell 3
# and no particles in cells 4,5,6,7
points = numpy.array([[-0.5, -0.5, -0.5],
                      [-0.6, -0.6, -0.6],
                      [0.5, -0.5, -0.5],
                      [-0.5, 0.5, -0.5],
                      [-0.6, 0.6, -0.5],
                      [-0.7, 0.7, -0.5]], dtype='float32')


box = trajectory.Box(2);
lc = locality.LinkCell(box, 1.0);
lc.computeCellList(points);

for i in xrange(0,6):
    print lc.getCell(points[i,:])

for c in range(0, lc.getNumCells()):
    cell_members = list(lc.itercell(c));
    print cell_members;

