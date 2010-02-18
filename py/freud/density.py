## \package freud.density
#
# Computes particle densities in various ways

import locality
import numpy
import math

## Compute g(r), also known as the radial distribution function
# TODO: document me
class RDF:
    ## Initialize the RDF compuataion
    # TODO: document me
    def __init__(self, box, rmax, dr):
        self.box = box;
        self.rmax = rmax;
        self.dr = dr;
        
        # allocate memory to hold the rdf
        self.nbins = int(math.ceil(self.rmax / self.dr));
        self.rdf_array = numpy.zeros(self.nbins, dtype='float32')
        self.bin_counts = numpy.zeros(self.nbins, dtype='uint32')
        
        # precompute the bin start positions
        self.r_array = numpy.zeros(self.nbins, dtype='float32')
        for i in xrange(0,self.nbins):
            self.r_array[i] = float(i) * dr;
        
        # precompute cell volumes
        self.vol_array = numpy.zeros(self.nbins, dtype='float32')
        self.vol_array[0] = 0.0;
        for i in xrange(1,self.nbins):
            r = float(i) * dr;
            self.vol_array[i] = 4.0 / 3.0 * math.pi * r**3.0 - self.vol_array[i-1];
        
        # create a link cell to bin the particles if the box is big enough
        if box.getLx() >= 3.0 * rmax and box.getLy() >= 3.0 * rmax and box.getLz() >= 3.0 * rmax:
            self.lc = locality.LinkCell(self.box, self.rmax);
        else:
            raise ValueError('RDF currently does not support computations where rmax > 1/3 any box dimension');
    
    ## Compute the rdf
    # \param x_ref x coordinates of reference points
    # \param y_ref y coordinates of reference points
    # \param z_ref z coordinates of reference points
    # \param x x coordinates of data points
    # \param y y coordinates of data points
    # \param z z coordinates of data points
    #
    # TODO: document me
    def compute(self, x_ref, y_ref, z_ref, x, y, z):
        # bin the x,y,z particles
        self.lc.computeCellList(x, y, z);
        
        # start by totalling up bin counts
        self.bin_counts[:] = 0;
        
        # for each reference point
        Nref = len(x_ref);
        for i in xrange(0, Nref):
            print "processing", i
            # get the cell the point is in
            ref_cell = self.lc.getCell(float(x_ref[i]), float(y_ref[i]), float(z_ref[i]));
            
            # loop over all neighboring cells
            neigh_cells = self.lc.getCellNeighbors(int(ref_cell));
            
            # for each neighboring cell
            for neigh_cell in neigh_cells:
                neigh_cell_particles = list(self.lc.itercell(int(neigh_cell)));
                
                # for each of the particles in neighboring cells
                for j in neigh_cell_particles:
                    # compute r between the two particles
                    dx = float(x_ref[i] - x[j]);
                    dy = float(y_ref[i] - y[j]);
                    dz = float(z_ref[i] - z[j]);
                    (dx, dy, dz) = self.box.wrap(dx, dy, dz);
                    
                    rsq = dx*dx + dy*dy + dz*dz;
                    r = math.sqrt(rsq);
                    
                    # bin that r
                    bin = int(math.floor(r / self.dr));
                    if bin < self.nbins:
                        self.bin_counts[bin] += 1;
                    
        # done looping over reference points
        # now compute the rdf
        normalize_factor = self.box.getVolume() / len(x);
        self.rdf_array = self.bin_counts / self.vol_array * normalize_factor;
        
        # done!
    
    ## Get a copy of the last computed RDF
    def getRDF(self):
        return self.rdf_array.copy();
    
    ## Get a copy of the array of r values
    def getR(self):
        return self.r_array.copy();
