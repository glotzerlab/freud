try:
    import VMD
except ImportError:
    VMD = None

import numpy

from _freud import Box;

## \package freud.trajectory
#
# Reads MD trajectories into numpy arrays
#
# TODO: complete a full set of documentation
# TODO: refactor common trajectory operations to a base class Trajectory
#
# The following classes are imported into trajectory from c++:
#  * Box

## \internal
# \brief Takes in a list of particle types and uniquely determines type ids for each one
# \note While typids will remain the same when run with the same types in the box, regardless of order, if one
# or another types are missing, then the typid assigments will differ. Thus, this is not to be used for trajectories
# where types vary from frame to frame and may dissapear
def _assign_typeid(typename):
    l = list(set(typename));
    l.sort();
    return [l.index(t) for t in typename];

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
            self.mol_id = VMD.molecule.get_top();

        self.mol = VMD.Molecule.Molecule(id=self.mol_id);
        self.all = VMD.atomsel.atomsel('all', molid=self.mol_id);

        # save the static properties
        self.static_props = {};
        self.static_props['mass'] = numpy.array(self.all.get('mass'), dtype='float32');
        self.static_props['diameter'] = 2.0*numpy.array(self.all.get('radius'), dtype='float32');
        self.static_props['typename'] = self.all.get('type');
        self.static_props['typeid'] = _assign_typeid(self.static_props['typename']);
        self.static_props['body'] = numpy.array(self.all.get('resid'), dtype=numpy.int32);
        self.static_props['charge'] = numpy.array(self.all.get('charge'), dtype=numpy.float32);
        
        self.modifiable_props = ['user', 'user2', 'user3', 'user4'];
        
    ## Test if a given particle property is modifiable
    # \param prop Property to check
    # \returns True if \a prop is modifiable
    def isModifiable(self, prop):
        return prop in self.modifiable_props;

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
    
    ## Sets the current frame
    # \param idx Index of the frame to seek to
    def setFrame(self, idx):
        self.mol.setFrame(idx);
    
    ## Get the current frame
    # \returns A FrameVMD containing the current frame data
    def getCurrentFrame(self):
        dynamic_props = {};

        # get position
        pos = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.float32);
        pos[:,0] = numpy.array(self.all.get('x'));
        pos[:,1] = numpy.array(self.all.get('y'));
        pos[:,2] = numpy.array(self.all.get('z'));
        dynamic_props['position'] = pos;
        
        # get user flags
        for prop in ['user', 'user2', 'user3', 'user4']:
            dynamic_props[prop] = numpy.array(self.all.get(prop), dtype=numpy.float32)
        
        vmdbox = VMD.molecule.get_periodic(self.mol_id, self.mol.curFrame())
        box = Box(vmdbox['a'], vmdbox['b'], vmdbox['c']);
        return Frame(self, self.mol.curFrame(), dynamic_props, box);
    
    ## Get the selected frame
    # \param idx Index of the frame to access
    # \returns A Frame containing the current frame data
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError('Frame index out of range');
        
        self.setFrame(idx);
        return self.getCurrentFrame();
    
    ## Iterate through frames
    def __iter__(self):
        return TrajectoryIter(self);
    
    ## Modify properties of the currently set frame
    # \param prop Name of property to modify
    # \param value New values to set for that property
    def setProperty(self, prop, value):
        # error check
        if not prop in self.modifiable_props:
            raise ValueError('prop is not modifiable');
        if len(value) != self.numParticles():
            raise ValueError('value is not of the correct length');
        
        self.all.set(prop, list(value));

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
    def __init__(self, traj, idx, dynamic_props, box):
        self.traj = traj;
        self.frame = idx;
        self.dynamic_props = dynamic_props;
        self.box = box;
    
    ## Access particle properties at this frame
    #
    # Particle properties are returned as numpy arrays with one element per particle. Properties are queried by name.
    # Common properties are
    #  - 'position' - particle positions (Nx3 array)
    #  - 'mass' - particle masses (Nx1 array)
    #
    # Some types of Trajectories may provide other properties. See their documentation for details.
    def get(self, prop):
        if prop in self.dynamic_props:
            return self.dynamic_props[prop];
        elif self.traj.isStatic(prop):
            return self.traj.getStatic(prop);
        else:
            raise KeyError('Particle property ' + prop + ' not found');
    
    ## Set properties for this frame
    # \param prop Name of property to modify
    # \param value New values to set for that property 
    #
    # Some types of properties can be set, depending on the Trajectory. For example, TrajectoryVMD allows setting of
    # the user, user2, user3, and user4 flags. Check if a property is modifiable with Trajectory.isModifiable()
    #
    # Some types of Trajectories may provide other modifiable properties. See their documentation for details.
    def set(self, prop, value):
        self.traj.setFrame(self.frame);
        self.traj.setProperty(prop, value);
    