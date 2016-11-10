# Proposed API Changes

To prepare for v1.0 and to fix a variety of design issues, here is a list of API changes (unordered)

# Indexing (specifically PMFT)

Address and impose common way of indexing arrays (PMFTR12 at least is in an odd order)

This really, really needs to be addressing because

## Numpy

myArray = np.zeros(shape=(nBinsZ, nBinsY, nBinsX))
val = myArray[z,y,x]

## Freud (C++)

Index3D myIndexer(nBinsX, nBinsY, nBinsz);
<T> val = myArray[myIndexer(x, y, z)];


# Gaussian Density

Change the constructor to take freud.density.GaussianDensity(width, r_cut, dr) always, and width is either a single
value OR a tuple. detect type, and move on with your life. This isn't nearly as bad as other API changes

# Function Verbosity

Set a standard for how verbose a function call should be:

    freud.locality.BondOrderDiagram()
    freud.locality.BOD()

SphericalHarmonicOrderParameter is the worst offender

# Allow for single position, orientation args

While all the code handles refPoints and points differently (which we probably need a new terminology for this too),
most users will probably want to check the positions against themselves, so we should allow:

    freud.module.submodule.compute(box, pos)
    freud.module.submodule.compute(box, refPos, pos)

# Change getRDF() to get?? in CorrelationFunction

# Change getFunction() Behavior

Currently reduce is called for each get, we should consider a switch that only reduces, etc. if necessary. The current
construction also prevents any python changes being back-populated to C++, for better or worse

## Update

so-called dirty flags have been implemented in parts of the code, and will be propagated throughout

# Allow users to pass in angles or quaternions to 2D calculations

Have the wrapper handle conversion. Might want to write as C lib
