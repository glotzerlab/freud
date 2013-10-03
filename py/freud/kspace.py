import numpy
import math
import copy
from math import *
from _freud import FTdelta as _FTdelta

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

## SingleCell3D objects manage data structures necessary to call the Fourier Transform functions that evaluate FTs for
# given form factors at a list of K points. SingleCell3D provides an interface to helper functions to calculate
# K points for a desired grid from the reciprocal lattice vectors calculated from an input boxMatrix.
# State is maintained as set_ and update_ functions invalidate internal data structures and as fresh data
# is restored with update_ function calls. This should facilitate management with a higher-level UI such as a GUI
# with an event queue.
#
# I'm not sure what sort of error checking would be most useful, so I'm mostly allowing ValueErrors and such exceptions
# to just occur and then propagate up through the calling functions to be dealt with by the user.
class SingleCell3D:
    ## Initialize the single-cell data structure for FT calculation
    # \param ndiv The resolution of the diffraction image grid
    # \param k The angular wave number of the plane wave probe. (Currently unused.)
    # \param dK The k-space unit associated with the diffraction image grid spacing
    # \param boxMatrix The unit cell lattice vectors as columns in a 3x3 matrix
    # \param scale nm per unit length (default 1.0)
    # The set_ functions take a single parameeter and cause other internal data structures to become invalid.
    # The update_ and calculate functions restore the validity of these structures using internal data.
    # The functions are separate to make it easier to avoid unnecessary computation such as when changing
    # multiple parameters before seeking output or when wrapping the code with an interface with an event queue.
    def __init__(self, k=1800, ndiv=16, dK=0.01, boxMatrix=None, *args, **kwargs):
        # Initialize some state
        self.Kpoints_valid = False
        self.FT_valid = False
        self.bases_valid = False
        self.K_constraint_valid = False

        # Set up particle type data structures
        self.ptype_name = list()
        self.ptype_position = list()
        self.ptype_orientation = list()
        self.ptype_ff = list()
        self.ptype_param = dict()
        self.ptype_param_methods = list()
        self.active_types = set()

        # Get arguments
        self.k = numpy.float32(k)
        self.ndiv = numpy.float32(ndiv)
        self.dK = numpy.float32(dK)
        if numpy.float32(boxMatrix).shape != (3,3):
            raise Warning('Need a valid box matrix!')
        else:
            self.boxMatrix = boxMatrix
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        else:
            self.scale = 1.0
        self.scale = numpy.float32(self.scale)
        self.a1, self.a2, self.a3 = numpy.zeros((3,3), dtype=numpy.float32)
        self.g1, self.g2, self.g3 = numpy.zeros((3,3), dtype=numpy.float32)
        self.update_bases()

        # Initialize remaining variables
        self.FT = None
        self.Kpoints = None
        self.Kmax = numpy.float32(self.dK * self.ndiv)
        K = self.Kmax
        R = K * numpy.float32(1.41422)
        epsilon = numpy.float32(-self.dK/2.)
        self.K_extent = [-K, K, -K, K, -epsilon, epsilon]
        self.K_constraint = None

        # For initial K points, assume a planar extent commensurate with the image
        self.update_K_constraint()
        self.update_Kpoints()

        # Set up particle type information and scattering form factor mapping
        self.fffactory = FTfactory()
        #self.fffactory.addFT('Sphere', FTsphere)

    ## Create internal data structures for new particle type by name
    # Particle type is inactive when added because parameters must be set before FT can be performed.
    # \param name particle name string
    def add_ptype(self, name):
        if name in self.ptype_name:
            raise Warning('{name} already exists'.format(name=name))
            return
        self.ptype_name.append(name)
        self.ptype_position.append(numpy.empty((0,3), dtype=numpy.float32))
        self.ptype_orientation.append(numpy.empty((0,4), dtype=numpy.float32))
        self.ptype_ff.append(None)
        for param in self.ptype_param:
            self.ptype_param[param].append(None)

    ## Remove internal data structures associated with ptype <name>
    # Note that this shouldn't usually be necessary, since particle types
    # may be set inactive or have any of their properties updated through set_ methods
    # \param name particle name string
    def remove_ptype(self, name):
        i = self.ptype_name.index(name)
        if i in self.active_types:
            self.active_types.remove(i)
        for param in self.ptype_param:
            self.ptype_param[param].remove(i)
        self.ptype_param_methods.remove(i)
        self.ptype_ff.remove(i)
        self.ptype_orientation.remove(i)
        self.ptype_position.remove(i)
        self.ptype_name.remove(i)
        if i in self.active_types:
            self.FT_valid = False

    ## Set particle type active
    # \param name particle name
    def set_active(self, name):
        i = self.ptype_name.index(name)
        if not i in self.active_types:
            self.active_types.add(i)
            self.FT_valid = False

    ## Set particle type inactive
    # \param name particle name
    def set_inactive(self, name):
        i = self.ptype_name.index(name)
        if i in self.active_types:
            self.active_types.remove(i)
            self.FT_valid = False

    ## Get ordered list of particle names
    def get_ptypes(self):
        return self.ptype_name
    ## Get form factor names and indices
    def get_form_factors(self):
        return self.fffactory.getFTlist()

    ## Set scattering form factor
    # \param name particle type name
    # \param ff scattering form factor named in self.get_form_factors()
    def set_form_factor(self, name, ff):
        i = self.ptype_name.index(name)
        j = self.fffactory.getFTlist().index(ff)
        FTobj = self.fffactory.getFTobject(j)
        # set FTobj parameters with previously chosen values
        for param in self.ptype_param:
            if param in FTobj.get_params():
                value = self.ptype_param[param][i]
                FTobj.set_parambyname(param, value)
        FTobj.set_parambyname('scale', self.scale)
        FTobj.set_rq(self.ptype_position[i], self.ptype_orientation[i])
        FTobj.set_K(self.Kpoints)
        self.ptype_ff[i] = FTobj
        if i in self.active_types:
            self.FT_valid = False

    ## Set named parameter for named particle
    # \param particle particle name
    # \param param parameter name
    # \param value parameter value
    def set_param(self, particle, param, value):
        i = self.ptype_name.index(particle)
        FTobj = self.ptype_ff[i]
        if not param in self.ptype_param:
            self.ptype_param[param] = [None] * len(self.ptype_name)
        self.ptype_param[param][i] = value
        if not param in FTobj.get_params():
            #raise KeyError('No set_ method for parameter {p}'.format(p=param))
            return
        else:
            FTobj.set_parambyname(param, value)
            if i in self.active_types:
                self.FT_valid = False
    ## Set scale factor. Store global value and set for each particle type
    # \param scale nm per unit for input file coordinates
    def set_scale(self, scale):
        self.scale = numpy.float32(scale)
        for i in xrange(len(self.ptype_ff)):
            self.ptype_ff[i].set_scale(scale)
        self.bases_valid = False

    ## Set positions and orientations for a particle type
    # To best maintain valid state in the event of changing numbers of particles, position and orienation are updated 
    # in a single method.
    # \param name particle type name
    # \param position (N,3) array of particle positions
    # \param orientation (N,4) array of particle quaternions
    def set_rq(self, name, position, orientation):
        i = self.ptype_name.index(name)
        r = numpy.asarray(position, dtype=numpy.float32)
        q = numpy.asarray(orientation, dtype=numpy.float32)
        # Check for compatible position and orientation
        N = r.shape[0]
        if q.shape[0] != N:
            raise ValueError('position and orientation must be the same length')
        if len(r.shape) != 2 or r.shape[1] != 3:
            raise ValueError('position must be a (N,3) array')
        if len(q.shape) != 2 or q.shape[1] != 4:
            raise ValueError('orientation must be a (N,4) array')
        self.ptype_position[i] = r
        self.ptype_orientation[i] = q
        self.ptype_ff[i].set_rq(r, q)
        if i in self.active_types:
            self.FT_valid = False

    ## Set number of grid divisions in diffraction image
    # \param ndiv define diffraction image as ndiv x ndiv grid
    def set_ndiv(self, ndiv):
        self.ndiv = int(ndiv)
        self.K_constraint_valid = False
    ## Set grid spacing in diffraction image
    # \param dK difference in K vector between two adjacent diffraction image grid points
    def set_dK(self, dK):
        self.dK = numpy.float32(dK)
        self.K_constraint_valid = False
    ## Set angular wave number of plane wave probe
    # \param k = |k_0|
    def set_k(self, k):
        self.k = numpy.float32(k)
        #self.K_points_valid = False
    ## Update the direct and reciprocal space lattice vectors
    # If scale or boxMatrix is updated, the lattice vectors in direct and reciprocal space need to be recalculated.
    def update_bases(self):
        self.bases_valid = True
        # Calculate scaled lattice vectors
        vectors = self.boxMatrix.transpose() * self.scale
        self.a1, self.a2, self.a3 = numpy.float32(vectors)
        # Calculate reciprocal lattice vectors
        self.g1, self.g2, self.g3 = numpy.float32(reciprocalLattice3D(self.a1, self.a2, self.a3))
        self.K_constraint_valid = False
    ## Recalculate constraint used to select K values
    # The constraint used is a slab of epsilon thickness in a plane perpendicular to the k0 propagation,
    # intended to provide easy emulation of TEM or relatively high-energy scattering.
    def update_K_constraint(self):
        self.K_constraint_valid = True
        self.Kmax = numpy.float32(self.dK * self.ndiv)
        K = self.Kmax
        R = K * numpy.float32(1.41422)
        epsilon = numpy.abs(numpy.float32(self.dK/2.))
        self.K_extent = [-K, K, -K, K, -epsilon, epsilon]
        self.K_constraint = AlignedBoxConstraint(R, *self.K_extent)
        self.Kpoints_valid = False
    ## Update K points at which to evaluate FT
    # If the diffraction image dimensions change relative to the reciprocal lattice,
    # the K points need to be recalculated. |K|=0 causes problems for some form factors,
    # so it is removed. If a direct scattering spot is desired, calculate and add it separately.
    # This can be fixed in the future...
    def update_Kpoints(self):
        self.Kpoints_valid = True
        self.Kpoints = numpy.float32(constrainedLatticePoints(self.g1, self.g2, self.g3, self.K_constraint))
        if len(self.Kpoints) > 0:
            truth = self.Kpoints == numpy.array([0.,0.,0.])
            nonzeros = numpy.invert(truth[:,0] * truth[:,1] * truth[:,2])
            self.Kpoints = self.Kpoints[nonzeros]
        for i in xrange(len(self.ptype_ff)):
            self.ptype_ff[i].set_K(self.Kpoints)
        self.FT_valid = False
    ## Calculate FT. The details and arguments will vary depending on the form factor chosen for the particles.
    # For any particle type-dependent parameters passed as keyword arguments, the parameter must be passed as a list
    # of length max(p_type)+1 with indices corresponding to the particle types defined. In other words, type-dependent
    # parameters are optional (depending on the set of form factors being calculated), but if included must be defined
    # for all particle types.
    # \param position (N,3) ndarray of particle positions in nm
    # \param orientation (N,4) ndarray of orientation quaternions
    # \param kwargs additional keyword arguments passed on to form-factor-specific FT calculator
    def calculate(self, *args, **kwargs):
        self.FT_valid = True
        shape = (len(self.Kpoints),)
        self.FT = numpy.zeros(shape, dtype=numpy.complex64)
        for i in self.active_types:
            calculator = self.ptype_ff[i]
            calculator.compute()
            self.FT += calculator.getFT()
        return self.FT

## Factory to return an FT object of the requested type
class FTfactory:
    def __init__(self):
        self.name_list = ['Delta']
        self.constructor_list = [FTdelta]
        self.args_list = [None]
    ## Get an ordered list of named FT types
    def getFTlist(self):
        return self.name_list
    ## Get a new instance of an FT type from list returned by getFTlist()
    # \param i index into list returned by getFTlist()
    # \param args argument object used to initialize FT, overriding default set at addFT()
    def getFTobject(self, i, args=None):
        constructor = self.constructor_list[i]
        if args is None:
            args = self.args_list[i]
        return constructor(args)
    ## Add an FT class to the factory
    # \param name identifying string to be returned by getFTlist()
    # \param constructor class / function name to be used to create new FT objects
    # \param args set default argument object to be used to construct FT objects
    def addFT(self, name, constructor, args=None):
        if name in self.name_list:
            raise Warning('{name} already in factory'.format(name=name))
        else:
            self.name_list.append(name)
            self.constructor_list.append(constructor)
            self.args_list.append(args)

## Base class for FT calculation classes
class FTbase:
    def __init__(self, *args, **kwargs):
        self.scale = numpy.float32(1.0)
        self.density = numpy.complex64(1.0)
        self.S = None
        self.K = numpy.array([[0., 0., 0.]], dtype=numpy.float32)
        self.position = None
        self.orientation = None

        # create dictionary of parameter names and set/get methods
        self.set_param_map = dict()
        self.set_param_map['scale'] = self.set_scale
        self.set_param_map['density'] = self.set_density

        self.get_param_map = dict()
        self.get_param_map['scale'] = self.get_scale
        self.get_param_map['density'] = self.get_density

    ## Compute FT
    #def compute(self, *args, **kwargs):
    def getFT(self):
        return self.S
    ## Get the parameter names accessible with set_parambyname()
    def get_params(self):
        return self.set_param_map.keys()
    ## Set named parameter for object
    # \param name parameter name. Must exist in list returned by get_params()
    # \param value parameter value to set
    def set_parambyname(self, name, value):
        if not name in self.set_param_map.keys():
            msg = 'Object {type} does not have parameter {param}'.format(type=self.__class__, param=name)
            raise KeyError(msg)
        else:
            self.set_param_map[name](value)
    ## Get named parameter for object
    # \param name parameter name. Must exist in list returned by get_params()
    def get_parambyname(self, name):
        if not name in self.get_param_map.keys():
            msg = 'Object {type} does not have parameter {param}'.format(type=self.__class__, param=name)
            raise KeyError(msg)
        else:
            return self.get_param_map[name]()
    ## Set K points to be evaluated
    # \param K list of K vectors at which to evaluate FT
    def set_K(self, K):
        self.K = numpy.asarray(K, dtype=numpy.float32)
    def set_scale(self, scale):
        self.scale = numpy.float32(scale)
    def get_scale(self):
        return self.scale
    def set_density(self, density):
        self.density = numpy.complex64(density)
    def get_density(self, density):
        return self.density
    def set_rq(self, r, q):
        self.position = numpy.asarray(r, dtype=numpy.float32)
        self.orientation = numpy.asarray(q, dtype=numpy.float32)
        if len(self.position.shape) == 1:
            self.position.resize((1,3))
        if len(self.position.shape) != 2:
            print('Error: can not make an array of 3D vectors from input position.')
            return None
        if len(self.orientation.shape) == 1:
            self.orientation.resize((1,4))
        if len(self.orientation.shape) != 2:
            print('Error: can not make an array of 4D vectors from input orientation.')
            return None

## Fourier transform a list of delta functions
class FTdelta(FTbase):
    def __init__(self, *args, **kwargs):
        FTbase.__init__(self)
        self.FTobj = _FTdelta()
    def set_K(self, K):
        FTbase.set_K(self, K)
        self.FTobj.set_K(self.K)
    def set_scale(self, scale):
        FTbase.set_scale(self, scale)
        self.FTobj.set_scale(float(self.scale))
    def set_density(self, density):
        FTbase.set_density(self, density)
        self.FTobj.set_density(complex(self.density))
    def set_rq(self, r, q):
        FTbase.set_rq(self, r, q)
        self.FTobj.set_rq(r, q)
    ## Compute FT
    # Calculate S = \sum_{\alpha} \exp^{-i \mathbf{K} \cdot \mathbf{r}_{\alpha}}
    def compute(self, *args, **kwargs):
        self.FTobj.compute()
        self.S = self.FTobj.getFT()

## Fourier transform a list of delta functions
class FTsphere(FTbase):
    def __init__(self, *args, **kwargs):
        FTbase.__init__(self, *args, **kwargs)
        self.set_param_map['radius'] = self.set_radius
        self.get_param_map['radius'] = self.get_radius
        self.radius = numpy.float32(0.5)
    ## Set radius parameter
    # \param radius sphere radius will be stored as given, but scaled by scale parameter when used by methods
    def set_radius(self, radius):
        self.radius = numpy.float32(radius)
    ## Get radius parameter
    # If appropriate, return value should be scaled by get_parambyname('scale') for interpretation.
    def get_radius(self):
        return self.radius
    ## Compute FT
    # Calculate $P(\mathbf{K}) = \sum_{\alpha} \rho_{\alpha}(\mathbf{r}) \exp^{-i \mathbf{K} \cdot \mathbf{r}_{\alpha}}$
    # For a set of uniform particles, the particle form factor is separable such that
    # $P(\mathbf{K}) = S(\mathbf{K}) F(\mathbf{K})
    # where S is the structure factor for a distribution of delta peaks and F is the form factor of the particle found
    # by Fourier transforming the particle scattering density in its local coordinates.
    def compute(self):
        radius = self.radius * self.scale
        position = self.position * self.scale
        self.outputShape = (self.K.shape[0],)
        Kmag2 = numpy.float32((self.K * self.K).sum(axis = -1))
        x = numpy.sqrt(Kmag2) * radius
        P = numpy.zeros(self.outputShape, dtype=numpy.float32)
        expKr = numpy.zeros(self.outputShape, dtype=numpy.complex64)
        for r in position:
            expKr += numpy.exp(numpy.dot(self.K, r) * -1.j)
            # P(K) = (4.*pi*R) / K**2 * (sinc(K*R) - cos(K*R)))
            # User should make sure |K| > 0 for all K
            # Note that numpy.sinc(x) == sin(pi * x)/(pi * x), while numpy.sin(x) == sin(x)
            # Shouldn't P be complex valued?
            P += (4. * numpy.pi * radius / Kmag2) * (numpy.sinc(x/numpy.pi) - numpy.cos(x))
        self.S = expKr * P * self.density

class FTconvexPolyhedron(FTbase):
    #! \param hull convex hull object as returned by freud.shape.ConvexPolyhedron(points)
    def __init__(self, hull, *args, **kwargs):
        FTbase.__init__(self, *args, **kwargs)
        self.set_param_map['radius'] = self.set_radius
        self.get_param_map['radius'] = self.get_radius
        self.hull = hull
    ## Set radius of in-sphere
    # \param radius inscribed sphere radius without scale applied
    def set_radius(self, radius):
        # Find original in-sphere radius, determine necessary scale factor, and scale vertices and surface distances
        inradius = abs(self.hull.equations[:, 3].max())
        scale_factor = radius / inradius
        self.hull.points *= scale_factor
        self.hull.equations[:,3] *= scale_factor
    ## Get radius of in-sphere
    # If appropriate, return value should be scaled by get_parambyname('scale') for interpretation.
    def get_radius(self):
        # Find current in-sphere radius
        inradius = abs(self.hull.equations[:,3].max())
        return inradius
    ## Compute FT
    # Calculate P = F * S
    # S = \sum_{\alpha} \exp^{-i \mathbf{K} \cdot \mathbf{r}_{\alpha}}
    # F is the analytical form factor for a polyhedron, computed with Spoly3D
    def compute(self, *args, **kwargs):
        # Return FT of delta function at one or more locations
        position = self.scale * self.position
        orientation = self.orientation
        self.outputShape = (self.K.shape[0],)
        self.S = numpy.zeros(self.outputShape, dtype=numpy.complex64)
        for r, q in zip(position, orientation):
            for i in xrange(len(self.K)):
                # The FT of an object with orientation q at a given k-space point is the same as the FT
                # of the unrotated object at a k-space point rotated the opposite way.
                # The opposite of the rotation represented by a quaternion is the conjugate of the quaternion,
                # found by inverting the sign of the imaginary components.
                K = quatrot(q * numpy.array([1,-1,-1,-1]), self.K[i])
                self.S[i] += numpy.exp(numpy.dot(K, r) * -1.j) * self.Spoly3D(K)
        self.S *= self.density
    ## Calculate Fourier transform of polygon
    # \param i face index into self.hull simplex list
    # \param k angular wave vector at which to calcular S(i)
    def Spoly2D(self, i, k):
        if numpy.dot(k, k) == 0.0:
            S = self.hull.getArea(i) * self.scale**2
        else:
            S = 0.0
            nverts = self.hull.nverts[i]
            verts = list(self.hull.facets[i, 0:nverts])
            # apply periodic boundary condition for convenience
            verts.append(verts[0])
            points = self.hull.points * self.scale
            n = self.hull.equations[i, 0:3]
            for j in xrange(self.hull.nverts[i]):
                v1 = points[verts[j+1]]
                v0 = points[verts[j]]
                edge = v1 - v0
                centrum = numpy.array((v1 + v0) / 2.)
                # Note that numpy.sinc(x) gives sin(pi*x)/pi*x
                x = numpy.dot(k, edge) / numpy.pi
                cpedgek = numpy.cross(edge, k)
                S += numpy.dot(n, cpedgek) * numpy.exp(-1.j * numpy.dot(k, centrum)) * numpy.sinc(x)
            S *= (-1.j / numpy.dot(k, k))
        return S
    ## Calculate Fourier transform of polyhedron
    # \param k angular wave vector at which to calculate FT.
    def Spoly3D(self, k):
        if numpy.dot(k, k) == 0.0:
            S = self.hull.getVolume() * self.scale**3
        else:
            S = 0.0
            # for face in faces
            for i in xrange(self.hull.nfacets):
                # need to project k into plane of face
                ni = self.hull.equations[i, 0:3]
                di = - self.hull.equations[i, 3] * self.scale
                dotkni = numpy.dot(k, ni)
                k_proj = k - ni * dotkni
                S += dotkni * numpy.exp(-1.j * dotkni * di) * self.Spoly2D(i, k_proj)
            S *= 1.j/(numpy.dot(k,k))
        return S

def mkSCcoords(nx, ny, nz):
    coords = list()
    for i in xrange(-int(nx/2), -int(nx/2) + nx):
        for j in xrange(-int(ny/2), -int(ny/2) + ny):
            for k in xrange(-int(nz/2), -int(nz/2) + nz):
                coords.append([i, j, k])
    return numpy.array(coords, dtype=float)
def mkBCCcoords(nx, ny, nz):
    # Note that now ni is number of half-lattice vectors
    coords = list()
    for i in xrange(-int(nx/2), -int(nx/2) + nx):
        for j in xrange(-int(ny/2), -int(ny/2) + ny):
            for k in xrange(-int(nz/2), -int(nz/2) + nz):
                if (i%2 == j%2) and (i%2 == k%2):
                    coords.append([i, j, k])
    return numpy.array(coords, dtype=float)
def mkFCCcoords(nx, ny, nz):
    # Note that now ni is number of half-lattice vectors
    coords = list()
    for i in xrange(-int(nx/2), -int(nx/2) + nx):
        for j in xrange(-int(ny/2), -int(ny/2) + ny):
            for k in xrange(-int(nz/2), -int(nz/2) + nz):
                if (i+j+k)%2 == 0:
                    coords.append([i, j, k])
    return numpy.array(coords, dtype=float)

## Axis angle rotation
# \param v vector to be rotated
# \param u rotation axis
# \param theta rotation angle
def rotate(v, u, theta):
    v = numpy.array(v) # need an actual array and not a view
    u = numpy.array(u)
    v.resize((3,))
    u.resize((3,))
    vx, vy, vz = v
    ux, uy, uz = u
    vout = numpy.empty((3,))
    st = sin(theta)
    ct = cos(theta)
    vout[0] = vx*(ct + ux*ux*(1 - ct))      \
            + vy*(ux*uy*(1 - ct) - uz*st)   \
            + vz*(ux*uz*(1 - ct) + uy*st)
    vout[1] = vx*(uy*ux*(1 - ct) + uz*st)    \
            + vy*(ct + uy*uy*(1-ct))        \
            + vz*(uy*uz*(1 - ct) - ux*st)
    vout[2] = vx*(uz*ux*(1 - ct) - uy*st)   \
            + vy*(uz*uy*(1 - ct) + ux*st)   \
            + vz*(ct + uz*uz*(1 - ct))
    return vout

## Apply a rotation quaternion
# \param b vector to be rotated
# \param a rotation quaternion
def quatrot(a, b):
    s = a[0]
    v = a[1:4]
    return (s*s - numpy.dot(v,v))*b + 2*s*numpy.cross(v,b) + 2*numpy.dot(v,b)*v

## Constraint base class
# Base class for constraints on vectors to define the API. All constraints should have a
# 'radius' defining a bounding sphere and a 'satisfies' method to determine whether an
# input vector satisfies the constraint.
class Constraint:
    ## Constructor
    # \param R required parameter describes the circumsphere of influence of the constraint for quick tests
    def __init__(self, R, *args, **kwargs):
        self.radius = R
    ## Constraint test
    # \param v vector to test against constraint
    def satisfies(self, v):
        return True

## Axis-aligned Box constraint
# Tetragonal box aligned with the coordinate system. Consider using a small z dimension
# to serve as a plane plus or minus some epsilon. Set R < L for a cylinder
class AlignedBoxConstraint(Constraint):
    def __init__(self, R, *args, **kwargs):
        self.radius = R
        self.R2 = R*R
        [self.xneg, self.xpos, self.yneg, self.ypos, self.zneg, self.zpos] = args
    def satisfies(self, v):
        satisfied = False
        if numpy.dot(v,v) <= self.R2:
            if v[0] >= self.xneg and v[0] <= self.xpos:
                if v[1] >= self.yneg and v[1] <= self.ypos:
                    if v[2] >= self.zneg and v[2] <= self.zpos:
                        satisfied = True
        return satisfied

## Generate a list of points satisfying a constraint
# \param v1,v2,v3 lattice vectors along which to test points
# \param constraint constraint object to test lattice points against
def constrainedLatticePoints(v1, v2, v3, constraint):
    # Find shortest distance, G, possible with lattice vectors
    # See how many G, nmax, fit in bounding box radius R
    # Limit indices h, k, l to [-nmax, nmax]
    # Check each value h, k, l to see if vector satisfies constraint
    # Return list of vectors
    R = constraint.radius
    R2 = R*R
    # Find shortest distance G. Assume lattice reduction is not necessary.
    gvec = v1 + v2 + v3
    G2 = numpy.dot(gvec, gvec)
    # This potentially checks redundant vectors, but optimization might require hard-to-unroll loops.
    for h in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            for l in [-1, 0, 1]:
                if [h, k, l] == [0, 0, 0]:
                    continue
                newvec = h*v1 + k*v2 + l*v3
                mag2 = numpy.dot(newvec, newvec)
                if mag2 < G2:
                    gvec = newvec
                    G2 = mag2
    G = numpy.sqrt(G2)
    nmax = int((R/G)+1)
    # Check each point against constraint
    # This potentially checks redundant vectors but we don't want to assume anything about the constraint.
    vec_list = list()
    for h in xrange(-nmax, nmax + 1):
        for k in xrange(-nmax, nmax + 1):
            for l in xrange(-nmax, nmax + 1):
                gvec = h*v1 + k*v2 + l*v3
                if constraint.satisfies(gvec):
                    vec_list.append(gvec)
    length = len(vec_list)
    vec_array = numpy.empty((length,3), dtype=numpy.float32)
    if length > 0:
        vec_array[...] = vec_list
    return vec_array

## Calculate reciprocal lattice vectors
# 3D reciprocal lattice vectors with magnitude equal to angular wave number
# \param a1,a2,a3 real space lattice vectors
# \returns list of reciprocal lattice vectors
def reciprocalLattice3D(a1, a2, a3):
    a1 = numpy.asarray(a1)
    a2 = numpy.asarray(a2)
    a3 = numpy.asarray(a3)
    a2xa3 = numpy.cross(a2, a3)
    g1 = (2 * numpy.pi / numpy.dot(a1, a2xa3)) * a2xa3
    a3xa1 = numpy.cross(a3, a1)
    g2 = (2 * numpy.pi / numpy.dot(a2, a3xa1)) * a3xa1
    a1xa2 = numpy.cross(a1, a2)
    g3 = (2 * numpy.pi / numpy.dot(a3, a1xa2)) * a1xa2
    return g1, g2, g3
# For unit test, note dot(g[i], a[j]) = 2 * pi * diracDelta(i, j)

## Base class for drawing diffraction spots on a 2D grid.
# Based on the dimensions of a grid, determines which grid points need to be modified to represent a diffraction spot
# and generates the values in that subgrid.
# Spot is a single pixel at the closest grid point
class DeltaSpot:
    ## Constructor
    # \param shape number of grid points in each dimension
    # \param extent range of x,y values associated with grid points
    def __init__(self, shape, extent, *args, **kwargs):
        self.shape = shape
        self.extent = extent
        self.dx = numpy.float32(extent[1] - extent[0]) / (shape[0] - 1)
        self.dy = numpy.float32(extent[3] - extent[2]) / (shape[1] - 1)
        self.x, self.y = numpy.float32(0), numpy.float32(0)
    ## Set x,y values of spot center
    # \param x x value of spot center
    # \param y y value of spot center
    def set_xy(self, x, y):
        self.x, self.y = numpy.float32(x), numpy.float32(y)
        # round to nearest grid point
        i = int(numpy.round((self.x - self.extent[0]) / self.dx))
        j = int(numpy.round((self.y - self.extent[2]) / self.dy))
        self.gridPoints = i, j
    ## Get indices of sub-grid
    # Based on the type of spot and its center, return the grid mask of points containing the spot
    def get_gridPoints(self):
        return self.gridPoints
    ## Generate intensity value(s) at sub-grid points
    # \param cval complex valued amplitude used to generate spot intensity
    def makeSpot(self, cval):
        return (numpy.conj(cval) * cval).real

## Draw diffraction spot as a Gaussian blur
# grid points filled according to gaussian at spot center
class GaussianSpot(DeltaSpot):
    def __init__(self, shape, extent, *args, **kwargs):
        DeltaSpot.__init__(self, shape, extent, *args, **kwargs)
        if 'sigma' in kwargs:
            self.set_sigma(kwargs['sigma'])
        else:
            self.set_sigma(self.dx)
        self.set_xy(0,0)
    ## Set x,y values of spot center
    # \param x x value of spot center
    # \param y y value of spot center
    def set_xy(self, x, y):
        self.x, self.y = numpy.float32(x), numpy.float32(y)
        # set grid: two index matrices of i and j values
        nx = int((3. * self.sigma / self.dx) + 1)
        ny = int((3. * self.sigma / self.dy) + 1)
        shape = (2*nx + 1, 2* ny + 1)
        gridx, gridy= numpy.indices(shape)
        # round center to nearest grid point
        i = int(numpy.round((self.x - self.extent[0]) / self.dx))
        j = int(numpy.round((self.y - self.extent[2]) / self.dy))
        gridx += i - nx
        gridy += j - ny
        # calculate x, y coordinates at grid points
        self.xvals = numpy.asarray(gridx * self.dx + self.extent[0], dtype=numpy.float32)
        self.yvals = numpy.asarray(gridy * self.dy + self.extent[2], dtype=numpy.float32)
        # remove values outside of extent
        mask =    (self.xvals >= self.extent[0]) \
                * (self.xvals <= self.extent[1]) \
                * (self.yvals >= self.extent[2]) \
                * (self.yvals <= self.extent[3])
        self.gridPoints = numpy.array([gridx[mask], gridy[mask]])
        self.xvals = self.xvals[mask]
        self.yvals = self.yvals[mask]
    ## Generate intensity value(s) at sub-grid points
    # \param cval complex valued amplitude used to generate spot intensity
    def makeSpot(self, cval):
        val = (numpy.conj(cval) * cval).real
        # calculate gaussian at grid points and multiply by val
        # currently assume "circular" gaussian: sigma_x = sigma_y
        # Precalculate gaussian argument
        x = self.xvals - self.x
        y = self.yvals - self.y
        gaussian = numpy.exp((-x * x - y * y) / (self.ss2))
        return val * gaussian
    ## Define Gaussian
    # \param sigma width of the Guassian spot
    def set_sigma(self, sigma):
        self.sigma = numpy.float32(sigma)
        self.ss2 = numpy.float32(sigma * sigma * 2)

# Not implemented due to lack of consensus on appropriate interpolation scheme
class InterpolatedDeltaSpot(DeltaSpot):
    # four grid points filled according to interpolation of delta at spot location
    def set_xy(self, x, y):
        self.x, self.y = x, y
        # set grid: two index matrices of i and j values
