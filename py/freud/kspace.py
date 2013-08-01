import numpy
import math
import copy

## \package freud.kspace
#
# Analyses that are compute quantities in kspace
#

## Computes an n-dimensional meshgrid
# \param arrs Arrays to grid
# source: http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = numpy.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans[::-1])


## Compute the full 3D structure factor of a given set of points
#
# Given a set of points \f$ \vec{r}_i \f$ SFactor3DPoints computes the static structure factor
# \f[ S(\vec{q}) = C_0 \left| \sum_{m=1}^{N} \exp(\mathit{i}\vec{q}\cdot\vec{r_i}) \right|^2 \f] where \f$ C_0 \f$ is a
# scaling constatnt chosen so that \f$ S(0) = 1 \f$, \f$ N \f$ is the number of particles. S is evaluated on a
# grid of q-values \f$ \vec{q} = h \frac{2\pi}{L_x} \hat{i} + k \frac{2\pi}{L_y} \hat{j} + l \frac{2\pi}{L_z} \hat{k} \f$
# for integer h,k,l ranging from -\a g to +\a g (inclusive) and \f$L_x, L_y, L_z \f$ are the box lengths
# in each direction.
#
# After calling compute(), access the used q values with getQ(), the static structure factor with getS(), and
# (if needed) the un-squared complex version of S with getSComplex(). All values are stored in 3D numpy arrays.
# They are indexed by a,b,c where a=h+g, b=k+g, and c=l+g.
#
# Note that due to the way that numpy arrays are indexed, access the returned S array as
# S[c,b,a] to get the value at q = (qx[a], qy[b], qz[c]).
class SFactor3DPoints:
    ## Initalize SFactor3DPoints:
    # \param box The simulation box
    # \param g The number of grid points for q in each direction is 2*g+1.
    def __init__(self, box, g):
        if box.is2D():
            raise ValueError("SFactor3DPoints does not support 2D boxes")

        self.grid = 2*g + 1;
        self.qx = numpy.linspace(-g * 2 * math.pi / box.getLx(), g * 2 * math.pi / box.getLx(), num=self.grid)
        self.qy = numpy.linspace(-g * 2 * math.pi / box.getLy(), g * 2 * math.pi / box.getLy(), num=self.grid)
        self.qz = numpy.linspace(-g * 2 * math.pi / box.getLz(), g * 2 * math.pi / box.getLz(), num=self.grid)

        # make meshgrid versions of qx,qy,qz for easy computation later
        self.qx_grid, self.qy_grid, self.qz_grid = meshgrid2(self.qx, self.qy, self.qz);

        # initialize a 0 S
        self.s_complex = numpy.zeros(shape=(self.grid,self.grid,self.grid), dtype=numpy.complex64);

    ## Compute the static structure factor of a given set of points
    # \param points Poits used to compute the static structure factor
    #
    # After calling compute(), you can access the results with getS(), getSComplex(), and the grid with getQ()
    def compute(self, points):
        # clear s_complex to zero
        self.s_complex[:,:,:] = 0;

        # add the contribution of each point
        for p in points:
            self.s_complex += numpy.exp(1j * (self.qx_grid * p[0] + self.qy_grid * p[1] + self.qz_grid * p[2]));

        # normalize
        mid = self.grid // 2;
        cinv = numpy.absolute(self.s_complex[mid,mid,mid]);
        self.s_complex /= cinv;

    ## Get the computed static structure factor
    # \returns The computed static structure factor as a copy
    def getS(self):
        return (self.s_complex * numpy.conj(self.s_complex)).astype(numpy.float32);

    ## Get the computed complex structure factor (if you need the phase information)
    # \returns The computed static structure factor, as a copy, without taking the magnitude squared
    def getSComplex(self):
        return copy.cpy(self.s_complex)

    ## Get the q values at each point
    # \returns qx, qy, qx
    # The structure factor S[c,b,c] is evaluated at the vector q = (qx[a], qy[b], qz[c])
    def getQ(self):
        return (self.qx, self.qy, self.qz);

## Analyze the peaks in a 3D structure factor
#
# Given a structure factor S(q) computed by classes such as SFactor3DPoints, AnalyzeSFactor3D
# performs a variety of analysis tasks.
#  - It identifies peaks
#  - Provides a list of peaks and the vector q positions at which they occur
#  - Provides a list of peaks grouped by q^2
#  - Provides a full list of S(|q|) values vs q^2 suitable for plotting the 1D analog of the structure factor
#  - Scans through the full 3d peaks and reconstructs the Bravais lattice
#
# Note that all of these operations work in an indexed integer q-space h,k,l. Any peak position values returned
# must be multiplied by 2*pi/L to to real q values in simulation units. (umm, need to think if this actually
# will work with non-cubic boxes....)
class AnalyzeSFactor3D:
    ## Initialize the analyzer
    # \param S Static structure factor to analyze
    #
    def __init__(self, S):
        self.S = S;
        self.grid = S.shape[0];
        self.g = self.grid/2;

    ## Get a list of peaks in the structure factor
    # \param cut All S(q) values greater than \a cut will be counted as peaks
    # \returns peaks, q as lists
    #
    def getPeakList(self, cut):
        clist,blist,alist = (self.S > cut).nonzero()
        clist -= self.g;
        blist -= self.g;
        alist -= self.g;

        q_list = [idx for idx in zip(clist,blist,alist)];
        peak_list = [self.S[(q[0]+self.g, q[1]+self.g, q[2]+self.g)] for q in q_list];
        return (q_list, peak_list);

    ## Get a dictionary of peaks indexed by q^2
    # \param cut All S(q) values greater than \a cut will be counted as peaks
    # \returns a dictionary with key q^2 and each element being a list of peaks
    #
    def getPeakDegeneracy(self, cut):
        q_list, peak_list = self.getPeakList(cut);

        retval = {}
        for q,peak in zip(q_list, peak_list):
            qsq = q[0]*q[0] + q[1] * q[1] + q[2] * q[2];
            if not (qsq in retval):
                retval[qsq] = [];

            retval[qsq].append(peak);

        return retval;

    # Get a list of all S(|q|) values vs q^2
    # \returns S, qsquared
    #
    def getSvsQ(self):
        hx = range(-self.g, self.g+1);

        qsq_list = [];
        # create an list of q^2 in the proper order
        for i in xrange(0,self.grid):
            for j in xrange(0,self.grid):
                for k in xrange(0,self.grid):
                    qsq_list.append(hx[i]*hx[i] + hx[j] * hx[j] + hx[k] * hx[k]);

        return (self.S.flatten(), qsq_list)


