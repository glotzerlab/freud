import freud
import itertools
import numpy

# create a simple cubic lattice
lattice = numpy.arange(10).astype(numpy.float32)
coordinates = numpy.array(list(itertools.product(lattice, lattice, lattice)))
coordinates -= numpy.mean(coordinates, axis=0)[numpy.newaxis, :]
coordinates += numpy.random.normal(scale=1e-3, size=coordinates.shape)

# make sure everything is inside the box
box = freud.trajectory.Box(len(lattice))
box.wrap(coordinates)

# grab all particles within rcut distance
rcut = 1.1

linkcell = freud.locality.LinkCell(box, rcut)
linkcell.computeCellList(box, coordinates)

# keep the contents of cells we've already used in a dictionary
cachedCellContents = {}

idx_i = []
idx_j = []

for (i, r_i) in enumerate(coordinates):
    myCell = linkcell.getCell(r_i)
    for neighborCell in linkcell.getCellNeighbors(myCell):
        try:
            neighbors = cachedCellContents[neighborCell]
        except KeyError:
            cachedCellContents[neighborCell] = list(linkcell.itercell(neighborCell))
            neighbors = cachedCellContents[neighborCell]
        idx_i.extend(len(neighbors)*[i])
        idx_j.extend(neighbors)
idx_i = numpy.array(idx_i, dtype=numpy.uint32)
idx_j = numpy.array(idx_j, dtype=numpy.uint32)

# rijs is the set of vectors from all particles to their neighbors
rijs = coordinates[idx_j] - coordinates[idx_i]
# apply periodic boundary conditions to vectors that span the boundary conditions
box.wrap(rijs)
rijsq = numpy.sum(rijs**2, axis=-1)

filtered = numpy.bitwise_and(rijsq > 1e-6, # exclude i-i interactions
                             rijsq < rcut**2) # exclude i-j interactions greater than rcut away

idx_i = idx_i[filtered]
idx_j = idx_j[filtered]

print('Bonds per particle: {}'.format(len(idx_i)/len(lattice)**3))
