try:
    import VMD
except ImportError:
    VMD = None

import numpy
import copy
import xml.dom.minidom

from _freud import Box;
import _freud;

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

## Base class Trajectory that defines a common interface for working with any trajectory
#
# TODO: Document me
#
class Trajectory:
    ## Initizlize an emtpy trajectory
    def __init__(self):
        self.static_props = {};
        self.modifiable_props = {};
    
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
    # \note The base class Trajectory doesn't load any particles, so this always returns 0. Derived classes
    #       should override.
    def numParticles(self):
        return 0;

    ## Get the number of frames in the trajectory
    # \returns Number of frames
    # \note The base class Trajectory doesn't load any particles, so this always returns 0. Derived classes
    #       should override.
    def __len__(self):
        return 0;
    
    ## Sets the current frame
    # \param idx Index of the frame to seek to
    # \note The base class Trajectory doesn't load any particles, so calling this method will produce an error.
    #       Derived classes should override
    def setFrame(self, idx):
        raise RuntimeError("Trajectory.setFrame not implemented");
    
    ## Get the current frame
    # \returns A Frame containing the current frame data
    # \note The base class Trajectory doesn't load any particles, so calling this method will produce an error.
    #       Derived classes should override
    def getCurrentFrame(self):
        raise RuntimeError("Trajectory.setFrame not implemented");

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
    # \note The base class Trajectory doesn't load any particles, so calling this method won't do anything.
    #       Derived classes can call it as a handy way to check for error conditions.
    def setProperty(self, prop, value):
        # error check
        if not prop in self.modifiable_props:
            raise ValueError('prop is not modifiable');
        if len(value) != self.numParticles():
            raise ValueError('value is not of the correct length');


## Trajectory information read directly from a running VMD instance
#
# TrajectoryVMD acts as a proxy to the VMD python data access APIs. It takes in a given molecule id and then presents
# a Trajectory interface on top of it, allowing looping through frames, accessing particle data an so forth.
#
# TrajectoryVMD only works when created inside of a running VMD instance. It will raise a RuntimeError if VMD is not
# found
#
class TrajectoryVMD(Trajectory):
    ## Initialize a VMD trajectory for access
    # \param mol_id Id number of the VMD molecule to access
    #
    # When \a mol_id is set to None, the 'top' molecule is accessed
    def __init__(self, mol_id=None):
        Trajectory.__init__(self);
    
        # check that VMD is loaded
        if VMD is None:
            raise RuntimeError('VMD is not loaded')

        # get the top molecule if requested
        if mol_id is None:
            self.mol_id = VMD.molecule.get_top();

        self.mol = VMD.Molecule.Molecule(id=self.mol_id);
        self.all = VMD.atomsel.atomsel('all', molid=self.mol_id);

        # save the static properties
        self.static_props['mass'] = numpy.array(self.all.get('mass'), dtype='float32');
        self.static_props['diameter'] = 2.0*numpy.array(self.all.get('radius'), dtype='float32');
        self.static_props['typename'] = self.all.get('type');
        self.static_props['typeid'] = _assign_typeid(self.static_props['typename']);
        self.static_props['body'] = numpy.array(self.all.get('resid'), dtype=numpy.int32);
        self.static_props['charge'] = numpy.array(self.all.get('charge'), dtype=numpy.float32);
        
        self.modifiable_props = ['user', 'user2', 'user3', 'user4'];
        
   
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
    
    ## Modify properties of the currently set frame
    # \param prop Name of property to modify
    # \param value New values to set for that property
    def setProperty(self, prop, value):
        # error check
        Trajectory.setProperty(self, prop, value);
        
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

## Trajectory information read from an XML/DCD file combination
#
# TrajectoryXMLDCD reads structure information in from the provided XML file (typenames, bonds, rigid bodies, etc...)
# Then, it opens the provided DCD file and reads in the position data for each frame from it.
#
# \note Always read DCD frames in increasing order for the best possible performance.
# While the Trajectory interface does allow for random access of specific frames, actually doing so
# is extremely inefficient for the DCD file format. To rewind to a previous frame, the file must be closed
# and every frame read from the beginning until the desired frame is reached! 
#
class TrajectoryXMLDCD(Trajectory):
    ## Initialize an XML/DCD trajectory for access
    # \param xml_fname File name of the XML file to read the structure from
    # \param dcd_fname File name of the DCD trajectory to read
    #
    def __init__(self, xml_fname, dcd_fname):
        Trajectory.__init__(self);
    
        # parse the XML file
        dom = xml.dom.minidom.parse(xml_fname);
        hoomd_xml = dom.getElementsByTagName('hoomd_xml');
        if len(hoomd_xml) != 1:
            raise RuntimeError("hoomd_xml tag not found in xml file")
        else:
            hoomd_xml = hoomd_xml[0];
            
        configuration = hoomd_xml.getElementsByTagName('configuration');
        if len(configuration) != 1:
            raise RuntimeError("configuration tag not found in xml file")
        else:
            configuration = configuration[0];

        # read the position node just to get the number of particles
        position = configuration.getElementsByTagName('position');
        if len(position) != 1:
            raise RuntimeError("hoomd_xml tag not found in xml file")
        else:
            position = position[0];
        position_text = position.childNodes[0].data
        xyz = position_text.split()
        self.num_particles = len(xyz)/3;

        # parse the particle types
        type_nodes = configuration.getElementsByTagName('type');
        if len(type_nodes) == 1:
            type_text = type_nodes[0].childNodes[0].data;
            type_names = type_text.split();
            if len(type_names) != self.num_particles:
                raise RuntimeError("wrong number of types found in xml file")
        else:
            raise RuntimeError("type tag not found in xml file")

        # parse the particle masses
        mass_nodes = configuration.getElementsByTagName('mass');
        if len(mass_nodes) == 1:
            mass_text = mass_nodes[0].childNodes[0].data;
            mass_list = mass_text.split();
            if len(mass_list) != self.num_particles:
                raise RuntimeError("wrong number of masses found in xml file")
            mass_array = numpy.array([float(m) for m in mass_list], dtype=numpy.float32);
        else:
            # default to a mass of 1.0, like hoomd
            mass_array = numpy.ones(shape=(1,self.num_particles), dtype=numpy.float32);
        
        # parse the particle diameters
        diam_nodes = configuration.getElementsByTagName('diameter');
        if len(diam_nodes) == 1:
            diam_text = diam_nodes[0].childNodes[0].data;
            diam_list = diam_text.split();
            if len(diam_list) != self.num_particles:
                raise RuntimeError("wrong number of diameters found in xml file")
            diameter_array = numpy.array([float(d) for d in diam_list], dtype=numpy.float32);
        else:
            # default to a diameter of 1.0, like hoomd
            diameter_array = numpy.ones(shape=(1,self.num_particles), dtype=numpy.float32);

        # parse the particle bodies
        body_nodes = configuration.getElementsByTagName('body');
        if len(body_nodes) == 1:
            body_text = body_nodes[0].childNodes[0].data;
            body_list = body_text.split();
            if len(body_list) != self.num_particles:
                raise RuntimeError("wrong number of bodies found in xml file")
            body_array = numpy.array([float(b) for b in body_list], dtype=numpy.int32);
        else:
            # default to a body of -1, like hoomd
            body_array = -1 * numpy.ones(shape=(1,self.num_particles), dtype=numpy.int32);

        # parse the particle charges
        charge_nodes = configuration.getElementsByTagName('charge');
        if len(charge_nodes) == 1:
            charge_text = charge_nodes[0].childNodes[0].data;
            charge_list = charge_text.split();
            if len(charge_list) != self.num_particles:
                raise RuntimeError("wrong number of charges found in xml file")
            charge_array = numpy.array([float(c) for c in charge_list], dtype=numpy.float32);
        else:
            # default to a charge of 0.0, like hoomd
            charge_array = numpy.zeros(shape=(1,self.num_particles), dtype=numpy.float32);

        # save the static properties
        self.static_props['mass'] = mass_array;
        self.static_props['diameter'] = diameter_array;
        self.static_props['typename'] = type_names;
        self.static_props['typeid'] = _assign_typeid(self.static_props['typename']);
        self.static_props['body'] = body_array;
        self.static_props['charge'] = charge_array;
        
        # load in the DCD file
        self.dcd_loader = _freud.DCDLoader(dcd_fname);
        if self.dcd_loader.getNumParticles() != self.num_particles:
            raise RuntimeError("number of particles in the DCD file doesn't match the number in the XML file");
   
    ## Get the number of particles in the trajectory
    # \returns Number of particles
    def numParticles(self):
        return self.num_particles;
    
    ## Get the number of frames in the trajectory
    # \returns Number of frames
    def __len__(self):
        return self.dcd_loader.getFrameCount();
    
    ## Sets the current frame
    # \param idx Index of the frame to seek to
    def setFrame(self, idx):
        self.dcd_loader.jumpToFrame(idx);
        self.dcd_loader.readNextFrame();
    
    ## Get the current frame
    # \returns A FrameVMD containing the current frame data
    def getCurrentFrame(self):
        dynamic_props = {};

        # get position
        pos = copy.copy(self.dcd_loader.getPoints());
        dynamic_props['position'] = pos;
        
        box = self.dcd_loader.getBox();

        return Frame(self, self.dcd_loader.getLastFrameNum(), dynamic_props, box);

## Trajectory information obtained from within a running hoomd instance
#
# TrajectoryHOOMD reads structure information and dynamic data in from a live, currently running hoomd simulation.
# In principle, much of the structure (i.e. particle types, bonds, etc...) could be changing as well. However,
# this first implementation will assume the most common case (that which TrajectoryXMLDCD and TrajectoryVMD follow)
# which is that mass, types, bonds, charges, and diameters remain constant and only position/velocity change from 
# step to step. These assumptions may be relaxed in a later version.
#
# TrajectoryHOOMD works a little different from the others in one respect. Because it is accessing the current
# state data directly from hoomd, there are no frames over which to loop. Accessing frame 0 will always return
# the current state of the system. Advancing forward of course must be done with hoomd run() commands.
#
class TrajectoryHOOMD(Trajectory):
    ## Initialize a HOOMD trajectory
    # \param sysdef System definition (returned from an init. call)
    #
    def __init__(self, sysdef):
        Trajectory.__init__(self);
        self.sysdef = sysdef;
    
        # extract "static" values
        mass_array = numpy.array([p.mass for p in self.sysdef.particles], dtype=numpy.float32);
        diameter_array = numpy.array([p.diameter for p in self.sysdef.particles], dtype=numpy.float32);
        # body_array = numpy.array([p.body for p in self.sysdef.particles], dtype=numpy.int32);
        charge_array = numpy.array([p.charge for p in self.sysdef.particles], dtype=numpy.float32);
        type_names = [p.type for p in self.sysdef.particles];
        typeid_array = numpy.array([p.typeid for p in self.sysdef.particles], dtype=numpy.uint32);

        # save the static properties
        self.static_props['mass'] = mass_array;
        self.static_props['diameter'] = diameter_array;
        self.static_props['typename'] = type_names;
        self.static_props['typeid'] = typeid_array;
        # self.static_props['body'] = body_array;  # unsupported in hoomd currently
        self.static_props['charge'] = charge_array;
        
    ## Get the number of particles in the trajectory
    # \returns Number of particles
    def numParticles(self):
        return len(self.sysdef.particles);
    
    ## Get the number of frames in the trajectory
    # \returns Number of frames
    def __len__(self):
        return 1;
    
    ## Sets the current frame
    # \param idx Index of the frame to seek to
    def setFrame(self, idx):
        pass;
    
    ## Get the current frame
    # \returns A FrameVMD containing the current frame data
    def getCurrentFrame(self):
        dynamic_props = {};

        # get position
        pos = numpy.array([p.position for p in self.sysdef.particles], dtype=numpy.float32);
        image = numpy.array([p.image for p in self.sysdef.particles], dtype=numpy.int32);
        velocity = numpy.array([p.velocity for p in self.sysdef.particles], dtype=numpy.float32);

        dynamic_props['position'] = pos;
        dynamic_props['image'] = image;
        dynamic_props['velocity'] = velocity;
        
        hoomd_box = self.sysdef.box;
        box = Box(hoomd_box[0], hoomd_box[1], hoomd_box[2]);

        return Frame(self, 0, dynamic_props, box);
