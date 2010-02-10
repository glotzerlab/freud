try:
    import VMD
except ImportError:
    VMD = None

import numpy

from _freud import Box;

## \package hoomd_analyze.trajectory
#
# Reads MD trajectories into numpy arrays
#
# TODO: complete a full set of documentation
# TODO: refactor common trajectory operations to a base class Trajectory
#
# The following classes are imported into trajectory from c++:
#  * Box

class TrajectoryIter:
    def __init__(self, traj):
        self.traj = traj;
        self.index = 0;
    def __iter__(self):
        return self;
    def next(self):
        if self.index == len(self.traj):
            raise StopIteration;
        
        result = self.traj[self.index];
        self.index += 1;
        return result;

## Trajectory information read directly from a running VMD instance
#
# TrajectoryVMD acts as a proxy to the VMD python data access APIs. It takes in a given molecule id and then presents
# a Trajectory interface on top of it, allowing looping through frames, accessing particle data an so forth.
#
# TrajectoryVMD only works when created inside of a running VMD instance. It will raise a RuntimeError if VMD is not
# found
#
class TrajectoryVMD:
    ## Initialize a VMD trajectory for access
    # \param mol_id Id number of the VMD molecule to access
    #
    # When \a mol_id is set to None, the 'top' molecule is accessed
    def __init__(self, mol_id=None):
        # check that VMD is loaded
        if VMD is None:
            raise RuntimeError('VMD is not loaded')

        # get the top molecule if requested
        if mol_id is None:
            mol_id = VMD.molecule.get_top();

        self.mol = VMD.Molecule.Molecule(id=mol_id);
        self.all = VMD.atomsel.atomsel('all', molid=mol_id);

        # save the static properties
        self.static_props = {};
        self.static_props['mass'] = numpy.array(self.all.get('mass'), dtype='float32');

    ## Test if a given particle property is static over the length of the trajectory
    # \param prop Property to check
    # \returns True if \a prop is static
    def isStatic(self, prop):
        return prop in self.static_props;
    
    ## Get a static property of the particles
    # \param prop Property name to get
    def getStatic(self, prop):
        return self.static_props[prop];
    
    ## Get the number of particles in the trajectory
    # \returns Number of particles
    def numParticles(self):
        return len(self.all);
    
    ## Get the number of frames in the trajectory
    # \returns Number of frames
    def __len__(self):
        return self.mol.numFrames();
    
    ## Get the current frame
    # \returns A FrameVMD containing the current frame data
    def getCurrentFrame(self):
        dynamic_props = {};
        for prop in ['x', 'y', 'z']:
            dynamic_props[prop] = numpy.array(self.all.get(prop), dtype='float32');
        
        return Frame(self, self.mol.curFrame(), dynamic_props);
    
    ## Get the current frame
    # \param idx Index of the frame to access
    # \returns A Frame containing the current frame data
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError('Frame index out of range');
        
        self.mol.setFrame(idx);
        return self.getCurrentFrame();
    
    ## Iterate through frames
    def __iter__(self):
        return TrajectoryIter(self);

## Frame information represneting the system state at a specific frame in a Trajectory
#
# High level classes should not construct Frame classes directly. Instead create a Trajectory and query it to get
# frames.
#
# Call access() to get the particle properties of the system at this frame. The member variable frame lists the current
# frame index.
#
class Frame:
    ## Initialize a frame for access
    # \param traj Parent Trajectory
    # \param idx Index of the frame
    # \param dynamic_props Dictionary of dynamic properties accessible in this frame
    #
    # \note  High level classes should not construct Frame classes directly. Instead create a Trajectory and query it 
    # to get frames
    def __init__(self, traj, idx, dynamic_props):
        self.traj = traj;
        self.frame = idx;
        self.dynamic_props = dynamic_props;
    
    ## Access particle properties at this frame
    #
    # Particle properties are returned as numpy arrays with one element per particle. Properties are queried by name.
    # Common properties are
    #  - 'x' - x coordiantes
    #  - 'y' - y coordiantes
    #  - 'z' - z coordiantes
    #  - 'mass' - particle masses
    #
    # Some types of Trajectories may provide other properties. See their documentation for details.
    def get(self, prop):
        if prop in self.dynamic_props:
            return self.dynamic_props[prop];
        elif self.traj.isStatic(prop):
            return self.traj.getStatic(prop);
        else:
            raise KeyError('Particle property ' + prop + ' not found');
    
    