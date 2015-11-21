# Proposed API Changes

To prepare for v1.0 and to fix a variety of design issues, here is a list of API changes (unordered)

# Remove Trajectory Module

This module has always felt out of place. With GlotzFormats being under development, all readers should be moved to it

# Change Location of Trajectory.Box

Whether or not the trajectory module is removed, Box does not belong in trajectory. Box is probably the most important
data structure in Freud and is utilized by nearly every module. After conversations with Eric and Richmond I propose the
following possibilities:

1. Move Box to its own module/namespace: Box
2. Move Box to util (as it can be viewed as a utility given that it does the work in terms of wrapping particles, etc.)
3. Move Box to a TBD module such as "data" (similar to hoomd). This would also be the location of other potential data
structures

# Change Box API

Regardless of the new location of Box, the way in which it is constructed needs to be improved. Currently any of the
following methods can be used to construct a box:

    freud.trajectory.Box(L)
    freud.trajectory.Box(L, is2D)
    freud.trajectory.Box(Lx, Ly, Lz)
    freud.trajectory.Box(Lx, Ly, is2D=False)
    freud.trajectory.Box(Lx=0.0, Ly=0.0, Lz=0.0, xy=0.0, xz=0.0, yz=0.0, is2D=False)

While the last method is preferred (and was the previous "new" API), it is a bit excessive if a user wants a 2D square
box. I propose the following possibilities:

1. Create unique constructor classes which inherit from the base Box class and correctly handle box construction:

        freud.box.SquareBox(L)
        freud.box.RectangularBox(Lx, Ly)
        freud.box.RhombicBox(Lx, Ly, xy)
        freud.box.CubicBox(L)
        freud.box.OrthorhombicBox(Lx, Ly, Lz)
        freud.box.TriclinicBox(Lx, Ly, Lz, xy, xz, yz)

2. Change the way in which we overload the python constructor (as overloading in python is not well defined).
This is not nearly as good as it's not clear the best way (at least right now) to handle the tilt-factor tuple.

        freud.box.Box(L, None, is2D)
        freud.box.Box(L, 0, is2D)
        freud.box.Box(L, (xy, xz, yz), is2D)
        freud.box.Box((Lx, Ly, Lz), (xy, xz, yz), is2D)

# Numpy Array Copy

Should the numpy arrays returned by calculations be copies of data, or pointers to the data itself? Both are easy to
implement. We could even add a copy "True/False" option. I have moved forward with including the copy option

# Indexing (specifically PMFT)

Address and impose common way of indexing arrays (PMFTR12 at least is in an odd order)

# Gaussian Density

Change the constructor to take freud.density.GaussianDensity(width, r_cut, dr) always, and width is either a single
value OR a tuple. detect type, and move on with your life. This isn't nearly as bad as other API changes

# Function Verbosity

Set a standard for how verbose a function call should be:

    freud.locality.BondOrderDiagram()
    freud.locality.BOD()

# Allow for single position, orientation args

While all the code handles refPoints and points differently (which we probably need a new terminology for this too),
most users will probably want to check the positions against themselves, so we should allow:

    freud.module.submodule.compute(box, pos)
    freud.module.submodule.compute(box, refPos, pos)

# Change getRDF() to get?? in CorrelationFunction

# Change getFunction() Behavior

Currently reduce is called for each get, we should consider a switch that only reduces, etc. if necessary

# Determine best way to return arrays

For any > 1D arrays determine if flat or reshaped arrays are the correct way to return to user

# Allow users to pass in angles or quaternions to 2D calculations

Have the wrapper handle conversion. Might want to write as C lib
