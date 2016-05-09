try:
    import VMD
except ImportError:
    VMD = None

import numpy
import copy
import threading
import xml.dom.minidom
try:
    import h5py
except ImportError:
    h5py = None

import warnings
warnings.simplefilter("always", DeprecationWarning)
warnings.warn("trajectory is slated for deprecation in v0.6.0. Please plan to use an external tool to load simulation \
 trajectories. Box will be its own module in v0.6.0.",
 DeprecationWarning)

from ._freud import Box
from ._freud import DCDLoader
from . import _freud
from freud.util import pos

## \internal
# \brief Takes in a list of particle types and uniquely determines type ids for each one
#
# \note While typids will remain the same when run with the same types in the box, regardless of order, if one
# or another types are missing, then the typid assigments will differ. Thus, this is not to be used for trajectories
# where types vary from frame to frame and may dissapear
def _assign_typeid(typename):
    l = list(set(typename))
    l.sort()
    return [l.index(t) for t in typename]

class Frame:
    """ Frame information representing the system state at a specific frame in a Trajectory.

    Initialize a frame for access. High level classes should not construct Frame classes directly. Instead create a Trajectory and query it to get frames.

    :param traj: Parent Trajectory
    :param idx: Index of the frame
    :param dynamic_props: Dictionary of dynamic properties accessible in this frame
    :param box: the simulation Box for this frame
    :type traj: :py:meth:`freud.trajectory.Trajectory`
    :type idx: int
    :type dynamic_props: list
    :type box: :py:meth:`freud.trajectory.Box`

    .. note:: High level classes should not construct Frame classes directly.
        Instead create a Trajectory and query it to get frames.

    Call get() to get the properties of the system at this frame. get() takes a string name of the property to be
    queried. If the property is static, the value of the property at the first frame will be returned from their
    Trajectory. If the property is dynamic, the value of the property at the current frame will be returned.

    Most properties are returned as numpy arrays. For example, position is returned as an Nx3 numpy array.
    """
    def __init__(self, traj, idx, dynamic_props, box, time_step=0):
        self.static_props = traj.static_props;
        self.frame = idx;
        self.dynamic_props = dynamic_props;
        self.box = box;
        self.time_step = time_step

    def get(self, prop):
        """ Access particle properties at this frame.

        Properties are queried by name. See the documentation of the specific Trajectory you load to see which
        properties it loads.

        """
        if prop in self.dynamic_props:
            return self.dynamic_props[prop];
        elif prop in self.static_props:
            return self.static_props[prop];
        else:
            raise KeyError('Particle property ' + prop + ' not found');

class Trajectory:
    """ Base class Trajectory that defines a common interface for working with any trajectory.

    A Trajectory represents a series of frames. Each frame consists of a set of properties on the particles, composite
    bodies, bonds, and walls in the system. Some trajectory formats may provide quantities that others do not.
    Some formats may provide certain properties only in the first frame, and others may store those properties at
    each frame. In addition, some formats may not even be capable of storing certain properties.

    Trajectory exposes properties as numpy arrays. Properties loaded for the first frame only are called static
    properties. The method isStatic() returns true if a given property is static.

    The Frame class provides access to the properties at any given frame. You can access a Frame by indexing a
    Trajectory directly::

        f = traj[frame_idx]

    Or by iterating over all frames::

        for f in traj:
            ...

    The number of frames in a trajectory is len(traj).

    .. note:: *Thread safety:*
        All trajectories provide thread-safe read only access to all parameters. Indexing f = traj[n] is serialized so that it
        can be performed in parallel by many threads.

    """
    def __init__(self):
        self.static_props = {}
        self.modifiable_props = {}
        self._lock = threading.Lock()

    def isModifiable(self, prop):
        """
        Test if a given particle property is modifiable.

        :param prop: Property to check
        :type prop: string
        :return: True if prop is modifiable
        :rtype: bool
        """
        return prop in self.modifiable_props

    def isStatic(self, prop):
        """
        Test if a given particle property is static over the length of the trajectory.

        :param prop: Property to check
        :type prop: string
        :return: True if prop is static
        :rtype: bool
        """
        return prop in self.static_props

    def getStatic(self, prop):
        """
        Get a static property of the particles.

        :param prop: Property name to get
        :type prop: string
        :return: property
        """
        return self.static_props[prop]

    def numParticles(self):
        """
        Get the number of particles in the trajectory.

        :return: Number of particles
        :rtype: int

        .. note:: The base class Trajectory doesn't load any particles, so this always returns 0.
            Derived classes should override.
        """
        return 0

    def __len__(self):
        """
        Get the number of frames in the trajectory.

        :return: Number of frames
        :rtype: int

        .. note::  The base class Trajectory doesn't load any frames, so this
        always returns 0. Derived classes should override.
        """
        return 0

    ## \internal
    # \brief Sets the current frame
    # \param idx Index of the frame to seek to
    # \note The base class Trajectory doesn't load any particles, so calling this method will produce an error.
    #       Derived classes should override
    def _set_frame(self, idx):
        raise RuntimeError("Trajectory._set_frame not implemented");

    ## \internal
    # \brief Get the current frame
    # \returns A Frame containing the current frame data
    # \note The base class Trajectory doesn't load any particles, so calling this method will produce an error.
    #       Derived classes should override
    def _get_current_frame(self):
        raise RuntimeError("Trajectory._get_current_frame not implemented");

    def __getitem__(self, idx):
        """
        Get the selected frame.

        :param idx: Index of the frame to access
        :type idx: int
        :return: A Frame containing the current frame data
        :rtype: :py:meth:`freud.trajectory.Frame`
        """
        if idx < 0 or idx >= len(self):
            raise IndexError('Frame index out of range');

        try:
            self._lock.acquire();
            self._set_frame(idx);
            return self._get_current_frame();
        finally:
            self._lock.release();

    def __iter__(self):
        """
        Iterate through the frames.
        """
        for idx in range(len(self)):
            yield self[idx];

    def setProperty(self, prop, value):
        """
        Modify properties of the currently set frame.

        :param prop: Name of property to modify
        :param value: New values to set for that property
        :type prop: string
        :type value: int, float, ...

        .. note:: The base class Trajectory doesn't load any particles, so calling this method won't do anything.
            Derived classes can call it as a handy way to check for error conditions.
        """
        # error check
        if not prop in self.modifiable_props:
            raise ValueError('prop is not modifiable');
        if len(value) != self.numParticles():
            raise ValueError('value is not of the correct length');

class TrajectoryVMD(Trajectory):
    """ Trajectory information read directly from a running VMD instance.

    TrajectoryVMD acts as a proxy to the VMD python data access APIs. It takes in a given molecule id and then presents
    a Trajectory interface on top of it, allowing looping through frames, accessing particle data an so forth.

    TrajectoryVMD only works when created inside of a running VMD instance. It will raise a RuntimeError if VMD is not
    found

    VMD has no way of specifiying 2D simulations explicitly. This code attempts to detect 2D simulations by checking
    the maximum z coord in the simulation. If the maximum z coord (in absolute value) is less than 1e-3, then the frame's
    box is set to 2D. If this is not what you intend, override the setting by calling box.set2D(True/False).

    :param mol_id: ID number of the vmd molecule to access. When a mol_id is set to None, the 'top' molecule is accessed.
    :type mol_id: int
    """
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

    def numParticles(self):
        """
        Get the number of particles in the trajectory.

        :return: Number of particles
        :rtype: int
        """
        return len(self.all);

    def __len__(self):
        """
        Get the number of frames in the trajectory.

        :return: Number of frames
        :rtype: int
        """
        return self.mol.numFrames();

    def _set_frame(self, idx):
        self.mol._set_frame(idx);

    def _get_current_frame(self):
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

        # detect if the box should be 2D
        if abs(pos[:,2]).max() < 1e-3:
            box.set2D(True);

        return Frame(self, self.mol.curFrame(), dynamic_props, box);

    def setProperty(self, prop, value):
        """
        Modify properties of the currently set frame.

        :param prop: Name of property to modify
        :param value: New values to set for that property
        :type prop: string
        :type value: int, float, ...
        """
        # error check
        Trajectory.setProperty(self, prop, value);

        self.all.set(prop, list(value));

class TrajectoryXML(Trajectory):
    """ Trajectory information read from a list of XML files.

    TrajectoryXML reads structure information in from the provided XML files (typenames, bonds, rigid bodies, etc...)
    storing each file as a consecutive frame.

    :param xml_fname_list: File names of the XML files to be read
    :param dynamic: List of dynamic properties in the trajectory
    :type xml_fname_list: list
    :type dynamic: list
    """
    def __init__(self, xml_fname_list, dynamic=['position']):
        Trajectory.__init__(self)

        # All properties implemented
        self.supported_props = ['position', 'image', 'velocity', 'acceleration', 'mass', 'diameter',
                                'charge', 'type', 'body', 'orientation', 'moment_inertia']
        for prop in dynamic:
            if prop not in self.supported_props:
                raise KeyError('Dynamic property "%s" not supported' % prop)

        self.xml_list = xml_fname_list

        # initialize dynamic list
        self.dynamic_props = {}
        for prop in dynamic:
            self.dynamic_props[prop] = {}

        # parse the initial XML file
        if len(xml_fname_list) == 0:
            raise RuntimeError("no filenames passed to TrajectoryXML")

        configuration = self._parseXML(self.xml_list[0])

        # determine the number of dimensions
        if configuration.hasAttribute('dimensions'):
            self.ndim = int(configuration.getAttribute('dimensions'))
        else:
            self.ndim = 3

        # read box
        box_config = configuration.getElementsByTagName('box')[0]
        xy = 0; xz = 0; yz = 0;
        if (box_config.hasAttribute('xy') and box_config.hasAttribute('xz') and box_config.hasAttribute('yz')):
            xy = float(box_config.getAttribute('xy'))
            xz = float(box_config.getAttribute('xz'))
            yz = float(box_config.getAttribute('yz'))

        self.box = Box(float(box_config.getAttribute('lx')),float(box_config.getAttribute('ly')),float(box_config.getAttribute('lz')),xy,xz,yz, self.ndim == 2)


        # Set the number of particles from the positions attribute
        position = configuration.getElementsByTagName('position')
        if len(position) != 1:
            raise RuntimeError("position tag not found in xml file")
        else:
            position = position[0]
        position_text = position.childNodes[0].data
        xyz = position_text.split()
        self.num_particles = int(len(xyz)/3)

        # Update the static properties if available in xml
        for prop in self.supported_props:
            # type has a special case
            if prop == "type": continue

            # Check if the xml has a data for the property
            if len(configuration.getElementsByTagName(prop)) == 1:
                prop_in_xml = True
            else:
                prop_in_xml = False

            if prop not in self.dynamic_props and prop_in_xml:
                self.static_props[prop] = self._update(prop, configuration)

        if 'type' not in self.dynamic_props and len(configuration.getElementsByTagName('type')) == 1:
            self.static_props['typename'] = self._update('type', configuration)
            self.static_props['typeid'] = _assign_typeid(self.static_props['typename'])

        self._set_frame(0)

    def numParticles(self):
        """
        Get the number of particles in the trajectory

        :return: Number of particles
        :rtype: int
        """
        return self.num_particles

    def __len__(self):
        """
        Get the number of frames in the trajectory

        :return: Number of frames
        :rtype: int
        """
        return len(self.xml_list)

    def _set_frame(self, idx):
        if idx >=  len(self.xml_list):
            raise RuntimeError("Invalid Frame Number")
        self.idx = idx

    def _get_current_frame(self):
        # load the information for the current frame
        configuration = self._parseXML(self.xml_list[self.idx])

        # Update box
        box_config = configuration.getElementsByTagName('box')[0]
        xy = 0; xz = 0; yz = 0;
        if (box_config.hasAttribute('xy') and box_config.hasAttribute('xz') and box_config.hasAttribute('yz')):
            xy = float(box_config.getAttribute('xy'))
            xz = float(box_config.getAttribute('xz'))
            yz = float(box_config.getAttribute('yz'))
        self.box = Box(float(box_config.getAttribute('lx')),float(box_config.getAttribute('ly')),float(box_config.getAttribute('lz')),xy,xz,yz, self.ndim == 2)

        # changed to add into dynamic_props as this would otherwise cause the for loop to barf
        if "type" in self.dynamic_props:
            self.dynamic_props['typename'] = self._update('type', configuration)
            self.dynamic_props['typeid'] = _assign_typeid(self.dynamic_props['typename'])
        for prop in self.dynamic_props.keys():
            if prop == 'typename' or prop == 'typeid':
                continue
            if prop == 'type':
                self.dynamic_props['typename'] = self._update('type', configuration)
                self.dynamic_props['typeid'] = _assign_typeid(self.dynamic_props['typename'])
            else:
                self.dynamic_props[prop] = self._update(prop, configuration)

        if configuration.hasAttribute('time_step'):
            curr_ts = int(configuration.getAttribute('time_step'))
        else:
            curr_ts = 0

        # copy.copy or copy.deepbox for self.box doesn't seem to be working, so declare a newBox object and return that, maybe changed later if the commented version works with other's modifications.
        newBox = Box(float(box_config.getAttribute('lx')),float(box_config.getAttribute('ly')),float(box_config.getAttribute('lz')),xy,xz,yz, self.ndim == 2)
        return Frame(self, self.idx, copy.deepcopy(self.dynamic_props), newBox, time_step = curr_ts)
        #return Frame(self, self.idx, copy.deepcopy(self.dynamic_props), copy.copy(self.box), time_step = curr_ts)



    def _parseXML(self, xml_filename):
        dom = xml.dom.minidom.parse(xml_filename)

        hoomd_xml = dom.getElementsByTagName('hoomd_xml')
        if len(hoomd_xml) != 1:
            raise RuntimeError("hoomd_xml tag not found in xml file")
        else:
            hoomd_xml = hoomd_xml[0]

        configuration = hoomd_xml.getElementsByTagName('configuration')
        if len(configuration) != 1:
            raise RuntimeError("configuration tag not found in xml file")
        else:
            return configuration[0]

    def _update(self, prop, configuration):
        # Data structure 3xFloatxNparticles
        if prop in ['position', 'velocity', 'acceleration']:
            raw_element = configuration.getElementsByTagName(prop)
            if len(raw_element) != 1:
                raise RuntimeError("%s tag not found in xml file" % prop)
            else:
                raw_data = raw_element[0]
            data_text = raw_data.childNodes[0].data
            xyz = data_text.split()

            data = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.float32)
            for i in range(0,self.num_particles):
                data[i,0] = float(xyz[3*i])
                data[i,1] = float(xyz[3*i+1])
                data[i,2] = float(xyz[3*i+2])
            return data

        # Data structure 3xIntxNparticles
        if prop in ['image']:
            raw_element = configuration.getElementsByTagName(prop)
            if len(raw_element) != 1:
                raise RuntimeError("%s tag not found in xml file" % prop)
            else:
                raw_data = raw_element[0]
            data_text = raw_data.childNodes[0].data
            xyz = data_text.split()

            data = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.int)
            for i in range(0,self.num_particles):
                data[i,0] = int(xyz[3*i])
                data[i,1] = int(xyz[3*i+1])
                data[i,2] = int(xyz[3*i+2])
            return data

        # Data structure: 1xFloatxNparticles
        if prop in ['mass', 'diameter','charge']:
            raw_element = configuration.getElementsByTagName(prop)
            if len(raw_element) != 1:
                raise RuntimeError("%s tag not found in xml file" % prop)
            else:
                raw_data = raw_element[0]
            data_text = raw_data.childNodes[0].data
            x = data_text.split()
            if len(x) != self.num_particles:
                raise RuntimeError("wrong number of %s values found in xml file" % prop)
            data = numpy.array([float(m) for m in x], dtype=numpy.float32)
            return data

         # Data structure: 1xIntxNparticles
        if prop in ['body']:
            raw_element = configuration.getElementsByTagName(prop)
            if len(raw_element) != 1:
                raise RuntimeError("%s tag not found in xml file" % prop)
            else:
                raw_data = raw_element[0]
            data_text = raw_data.childNodes[0].data
            x = data_text.split()
            if len(x) != self.num_particles:
                raise RuntimeError("wrong number of %s values found in xml file" % prop)
            data = numpy.array([int(m) for m in x], dtype=numpy.int)
            return data

        # Data structure: 4xFloatxNparticles
        if prop in ['orientation']:
            raw_element = configuration.getElementsByTagName(prop)
            if len(raw_element) != 1:
                raise RuntimeError("%s tag not found in xml file" % prop)
            else:
                raw_data = raw_element[0]
            data_text = raw_data.childNodes[0].data
            quat = data_text.split()

            data = numpy.zeros(shape=(self.numParticles(),4), dtype=numpy.float32)
            for i in range(0,self.num_particles):
                for j in range(4):
                    data[i,j] = float(quat[4*i+j])
            return data

        # Data structure: 6xIntxNparticles
        if prop in ['moment_inertia']:
            raw_element = configuration.getElementsByTagName(prop)
            if len(raw_element) != 1:
                raise RuntimeError("%s tag not found in xml file" % prop)
            else:
                raw_data = raw_element[0]
            data_text = raw_data.childNodes[0].data
            quat = data_text.split()

            data = numpy.zeros(shape=(self.numParticles(),6), dtype=numpy.float32)
            for i in range(0,self.num_particles):
                for j in range(6):
                    data[i,j] = float(quat[6*i+j])
            return data


        if prop == 'type':
            type_nodes = configuration.getElementsByTagName('type')
            if len(type_nodes) == 1:
                type_text = type_nodes[0].childNodes[0].data
                type_names = type_text.split()
                if len(type_names) != self.num_particles:
                    raise RuntimeError("wrong number of types found in xml file")
            else:
                raise RuntimeError("type tag not found in xml file")
            return type_names

class TrajectoryXMLDCD(Trajectory):
    """ Trajectory information read from an XML/DCD file combination.

    TrajectoryXMLDCD reads structure information in from the provided XML file (typenames, bonds, rigid bodies, etc...)
    Then, it opens the provided DCD file and reads in the position data for each frame from it.

    .. note:: Always read DCD frames in increasing order for the best possible performance.
        While the Trajectory interface does allow for random access of specific frames, actually doing so
        is extremely inefficient for the DCD file format. To rewind to a previous frame, the file must be closed
        and every frame read from the beginning until the desired frame is reached!

    2D input will set the frame box appropriately.

    :param xml_fname: File name of the XML file to read the structure from
    :param dcd_fname: File name of the DCD trajectory to read (or None to skip reading the trajectory data)
    :type xml_fname: string
    :type dcd_fname: string
    """
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

        # determine the number of dimensions
        if configuration.hasAttribute('dimensions'):
            self.ndim = int(configuration.getAttribute('dimensions'));
        else:
            self.ndim = 3;


        # (Chrisy: seems not possible to reach this if statement since the initialization requires a dcd file)
        # if there is no dcd file, read box
        if dcd_fname is None:

            box_config = configuration.getElementsByTagName('box')[0]
            xy = 0; xz = 0; yz = 0;
            if (box_config.hasAttribute('xy') and box_config.hasAttribute('xz') and box_config.hasAttribute('yz')):
                xy = float(box_config.getAttribute('xy'))
                xz = float(box_config.getAttribute('xz'))
                yz = float(box_config.getAttribute('yz'))

            self.box = Box(float(box_config.getAttribute('lx')),float(box_config.getAttribute('ly')),float(box_config.getAttribute('lz')),xy,xz,yz, self.ndim == 2)

        # read the position node just to get the number of particles
        # unless there is no dcd file. Then read positions.
        position = configuration.getElementsByTagName('position');
        if len(position) != 1:
            raise RuntimeError("hoomd_xml tag not found in xml file")
        else:
            position = position[0];
        position_text = position.childNodes[0].data
        xyz = position_text.split()
        self.num_particles = int(len(xyz)/3)
        if dcd_fname is None:
            pos = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.float32);
            for i in range(0,self.num_particles):
                pos[i,0] = float(xyz[3*i]);
                pos[i,1] = float(xyz[3*i+1]);
                pos[i,2] = float(xyz[3*i+2]);

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
        if dcd_fname is None:
            self.static_props['position'] = pos;


        # load in the DCD file
        if dcd_fname is not None:
            self.dcd_loader = _freud.DCDLoader(dcd_fname);
            if self.dcd_loader.getNumParticles() != self.num_particles:
                raise RuntimeError("number of particles in the DCD file doesn't match the number in the XML file");
        else:
            self.dcd_loader = None;

    def numParticles(self):
        """
        Get the number of particles in the trajectory.

        :return: Number of particles
        :rtype: int
        """
        return self.num_particles;

    def __len__(self):
        """
        Get the number of frames in the trajectory.

        :return: Number of frames
        :rtype: int
        """
        if self.dcd_loader is not None:
            return self.dcd_loader.getFrameCount();
        else:
            return 1;

    def _set_frame(self, idx):
        if self.dcd_loader is None and idx > 0:
            raise RuntimeError("No DCD file was loaded");

        if self.dcd_loader is not None and not(self.dcd_loader.getLastFrameNum() == idx):
            self.dcd_loader.jumpToFrame(idx);
            self.dcd_loader.readNextFrame();

    def _get_current_frame(self):

        dynamic_props = {};

        # get position
        if self.dcd_loader is not None:
            pos = self.dcd_loader.getPoints();
            dynamic_props['position'] = pos;
            dcdBox = self.dcd_loader.getBox();
            newBox = Box(dcdBox.getLx(),
                         dcdBox.getLy(),
                         dcdBox.getLz(),
                         dcdBox.getTiltFactorXY(),
                         dcdBox.getTiltFactorXZ(),
                         dcdBox.getTiltFactorYZ(),
                         self.ndim == 2)
            return Frame(self,
                         self.dcd_loader.getLastFrameNum(),
                         dynamic_props,
                         newBox,
                         time_step=self.dcd_loader.getTimeStep());
        else:
            box = self.box;
            return Frame(self, 1, dynamic_props, box);

class TrajectoryPOS(Trajectory):
    """ Trajectory information read from an POS file.

    TrajectoryPOS reads structure information in from the provided POS file.

    :param pos_fname: File name of the POS file to read the structure from
    :param dynamic: List of dynamic properties in the trajectory
    :type pos_fname: string
    :param dynamic: list
    """
    def __init__(self, pos_fname, dynamic=['boxMatrix', 'position', 'orientation']):
        Trajectory.__init__(self);

        self.dynamic_props = {}
        for prop in dynamic:
            self.dynamic_props[prop] = {}

        # parse the POS file
        self.pos_file = pos.file(pos_fname);
        self.pos_file.grabBox();

        # Is there a place in the pos that specs the dims?
        dim_test = len(self.pos_file.box_positions[0][0])
        if dim_test == 2:
            self.ndim = 2;
        else:
            self.ndim = 3

        box_dims = numpy.asarray(self.pos_file.box_dims[0], dtype=numpy.float32)
        if len(box_dims) == 3:
            lx = box_dims[0];
            ly = box_dims[1];
            lz = box_dims[2];
            xy = 0;
            xz = 0;
            yz = 0;
        else:
            # Store original box matrix as a static property.
            e1 = box_dims[[0,3,6]]
            e2 = box_dims[[1,4,7]]
            e3 = box_dims[[2,5,8]]
            if not 'boxMatrix' in self.dynamic_props:
                self.static_props['boxMatrix'] = numpy.asarray([e1, e2, e3]).transpose()
            lx = numpy.sqrt(numpy.dot(e1, e1))
            a2x = numpy.dot(e1, e2) / lx
            ly = numpy.sqrt(numpy.dot(e2,e2) - a2x*a2x)
            xy = a2x / ly
            v0xv1 = numpy.cross(e1, e2)
            v0xv1mag = numpy.sqrt(numpy.dot(v0xv1, v0xv1))
            lz = numpy.dot(e3, v0xv1) / v0xv1mag
            a3x = numpy.dot(e1, e3) / lx
            xz = a3x / lz
            yz = (numpy.dot(e2,e3) - a2x*a3x) / (ly*lz)

            # Enlarge tetragonal box to include all of triclinic box. It would be best if
            # the box matrix were upper triangular and right-handed.
            #diagonal = e1 + e2 + e3
            #box_vecs = numpy.asarray([e1, e2, e3, [0, 0, 0], diagonal])
            #lx = box_vecs[:,0].max() - box_vecs[:,0].min()
            #ly = box_vecs[:,1].max() - box_vecs[:,1].min()
            #lz = box_vecs[:,2].max() - box_vecs[:,2].min()
        #print("lx = {0} ly = {1} lz = {2}".format(*box_dims))
        self.box = Box(float(lx), float(ly), float(lz), float(xy), float(xz), float(yz), self.ndim == 2);

        #Reader can handle changing num particles, but this doesn't
        self.num_particles = int(self.pos_file.n_box_points[0])

        # Update the static properties
        if not 'boxMatrix' in self.dynamic_props:
            self.static_props['boxMatrix'] = self._update('boxMatrix', 0)
        if not 'position' in self.dynamic_props:
            self.static_props['position'] = self._update('position', 0)
        if not 'orientation' in self.dynamic_props:
            self.static_props['orientation'] = self._update('orientation', 0)
        #if not 'velocity' in self.dynamic_props:
        #   self.static_props['velocity'] = self._update('velocity', configuration)
        #if not 'mass' in self.dynamic_props:
        #    self.static_props['mass'] = self._update('mass', configuration)
        #if not 'diameter' in self.dynamic_props:
        #    self.static_props['diameter'] = self._update('diameter', configuration)
        if not 'type' in self.dynamic_props:
            self.static_props['typename'] = self._update('type', 0)
            self.static_props['typeid'] = _assign_typeid(self.static_props['typename'])
        #if not 'body' in self.dynamic_props:
        #    self.static_props['body'] = self._update('body', configuration)
        #if not 'charge' in self.dynamic_props:
        #    self.static_props['charge'] = self._update('charge', configuration)
        self.setFrame(0)

    def numParticles(self):
        """
        Get the number of particles in the trajectory

        :return: Number of particles
        :rtype: int
        """
        return self.num_particles;

    def __len__(self):
        """
        Get the number of frames in the trajectory

        :return: Number of frames
        :rtype: int
        """
        return len(self.pos_file.box_positions)

    ## Sets the current frame
    # \param idx Index of the frame to seek to
    def _set_frame(self, idx):
        # Does this offset the frame by 1?
        if idx >=  len(self.pos_file.box_positions):
            raise RuntimeError("Invalid Frame Number")
        self.idx = idx
    def setFrame(self, idx):
        return self._set_frame(idx)

    ## Get the current frame
    # \returns A Frame containing the current frame data
    def _get_current_frame(self):
        dynamic_props = {};
        # get position
        for prop in self.dynamic_props.keys():
            self.dynamic_props[prop] = self._update(prop, self.idx)

        return Frame(self, self.idx, self.dynamic_props, self.box)
    def getCurrentFrame(self):
        return self._get_current_frame()

    def _update(self, prop, frame_number):
        if prop == 'boxMatrix':
            box_dims = numpy.asarray(self.pos_file.box_dims[frame_number], dtype=numpy.float32)
            # Changed to support box and boxmatrix...
            # Could be handled in another way
            if len(box_dims) == 3:
                lx = box_dims[0];
                ly = box_dims[1];
                lz = box_dims[2];
                xy = 0;
                xz = 0;
                yz = 0;
            else:
                # This whole bit is kludgey and in need of some standardization.
                # Store original box matrix as a static property.
                e1 = box_dims[[0,3,6]]
                e2 = box_dims[[1,4,7]]
                e3 = box_dims[[2,5,8]]
                if not 'boxMatrix' in self.dynamic_props:
                    self.static_props['boxMatrix'] = numpy.asarray([e1, e2, e3]).transpose()
                lx = numpy.sqrt(numpy.dot(e1, e1))
                a2x = numpy.dot(e1, e2) / lx
                ly = numpy.sqrt(numpy.dot(e2,e2) - a2x*a2x)
                xy = a2x / ly
                v0xv1 = numpy.cross(e1, e2)
                v0xv1mag = numpy.sqrt(numpy.dot(v0xv1, v0xv1))
                lz = numpy.dot(e3, v0xv1) / v0xv1mag
                a3x = numpy.dot(e1, e3) / lx
                xz = a3x / lz
                yz = (numpy.dot(e2,e3) - a2x*a3x) / (ly*lz)
            self.box = Box(float(lx), float(ly), float(lz), float(xy), float(xz), float(yz), self.ndim == 2);
        if prop == 'position':
            position = self.pos_file.box_positions[frame_number]
            #if len(position) != 1:
            #    raise RuntimeError("position tag not found in xml file")
            #else:
            #    position = position[0]
            #position_text = position.childNodes[0].data
            #xyz = position_text.split()

            pos = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.float32)
            for i in range(0,self.num_particles):
                pos[i,0] = float(position[i][0])
                pos[i,1] = float(position[i][1])
                pos[i,2] = float(position[i][2])
            return pos

        if prop == 'orientation':
            orientation = self.pos_file.box_orientations[frame_number]
            #if len(position) != 1:
            #    raise RuntimeError("position tag not found in xml file")
            #else:
            #    position = position[0]
            #position_text = position.childNodes[0].data
            #xyz = position_text.split()

            quat = numpy.zeros(shape=(self.numParticles(),4), dtype=numpy.float32)
            for i in range(0,self.num_particles):
                quat[i,0] = float(orientation[i][0])
                quat[i,1] = float(orientation[i][1])
                quat[i,2] = float(orientation[i][2])
                quat[i,3] = float(orientation[i][3])
            return quat

        if prop == 'type':
            #type_nodes = configuration.getElementsByTagName('type')
            #if len(type_nodes) == 1:
            #    type_text = type_nodes[0].childNodes[0].data
            #    type_names = type_text.split()
            #    if len(type_names) != self.num_particles:
            #        raise RuntimeError("wrong number of types found in xml file")
            #else:
            #    raise RuntimeError("type tag not found in xml file")
            type_key = self.pos_file.type_names[frame_number]
            type_ID = self.pos_file.types[frame_number]
            type_names = []
            for i in type_ID:
                type_names.append(type_key[i])
            return type_names

        # Need to have an error raised for mass cause I don't think that mass is here

        #if prop == 'mass':
        #    mass_nodes = configuration.getElementsByTagName('mass')
        #    if len(mass_nodes) == 1:
        #        mass_text = mass_nodes[0].childNodes[0].data
        #        mass_list = mass_text.split()
        #        if len(mass_list) != self.num_particles:
        #            raise RuntimeError("wrong number of masses found in xml file")
        #        mass_array = numpy.array([float(m) for m in mass_list], dtype=numpy.float32)
        #    else:
                # default to a mass of 1.0, like hoomd
        #        mass_array = numpy.ones(shape=(1,self.num_particles), dtype=numpy.float32)
        #        return mass_array

        # This is going to be complicated because this will involve shape def...
        #if prop == 'diameter':
        #   diam_nodes = configuration.getElementsByTagName('diameter')
        #    if len(diam_nodes) == 1:
        #        diam_text = diam_nodes[0].childNodes[0].data
        #        diam_list = diam_text.split()
        #        if len(diam_list) != self.num_particles:
        #            raise RuntimeError("wrong number of diameters found in xml file")
        #        diameter_array = numpy.array([float(d) for d in diam_list], dtype=numpy.float32)
        #    else:
                # default to a diameter of 1.0, like hoomd
        #        diameter_array = numpy.ones(shape=(1,self.num_particles), dtype=numpy.float32)
        #    return diameter_array

        #if prop == 'body':
        #    body_nodes = configuration.getElementsByTagName('body')
        #    if len(body_nodes) == 1:
        #        body_text = body_nodes[0].childNodes[0].data
        #        body_list = body_text.split()
        #        if len(body_list) != self.num_particles:
        #            raise RuntimeError("wrong number of bodies found in xml file")
        #        body_array = numpy.array([float(b) for b in body_list], dtype=numpy.int32)
        #    else:
                # default to a body of -1, like hoomd
        #        body_array = -1 * numpy.ones(shape=(1,self.num_particles), dtype=numpy.int32)
        #    return body_array

        #if prop == 'charge':
        #    charge_nodes = configuration.getElementsByTagName('charge')
        #    if len(charge_nodes) == 1:
        #        charge_text = charge_nodes[0].childNodes[0].data
        #        charge_list = charge_text.split()
        #        if len(charge_list) != self.num_particles:
        #            raise RuntimeError("wrong number of charges found in xml file")
        #        charge_array = numpy.array([float(c) for c in charge_list], dtype=numpy.float32)
        #    else:
                # default to a charge of 0.0, like hoomd
        #        charge_array = numpy.zeros(shape=(1,self.num_particles), dtype=numpy.float32)
        #    return charge_array

        #if prop == 'velocity':
        #    velocity = configuration.getElementsByTagName('velocity')
        #    if len(velocity) == 1:
        #        velocity = velocity[0]
        #        velocity_text = velocity.childNodes[0].data
        #        xyz = velocity_text.split()
        #        if len(xyz)/3 != self.num_particles:
        #            raise RuntimeError("wrong number of velocities found in xml file")

        #        velocity_array = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.float32)
        #        for i in range(0,self.num_particles):
        #            velocity_array[i,0] = float(xyz[3*i])
        #            velocity_array[i,1] = float(xyz[3*i+1])
        #            velocity_array[i,2] = float(xyz[3*i+2])
        #    else:
                # default to zero
        #        velocity_array = numpy.zeros(shape=(self.numParticles(),3), dtype=numpy.float32)
        #    return velocity_array

class TrajectoryHOOMD(Trajectory):
    """
    Trajectory information obtained from within a running hoomd instance

    TrajectoryHOOMD reads structure information and dynamic data in from a live, currently running hoomd simulation.
    In principle, much of the structure (i.e. particle types, bonds, etc...) could be changing as well. However,
    this first implementation will assume the most common case (that which TrajectoryXMLDCD and TrajectoryVMD follow)
    which is that mass, types, bonds, charges, and diameters remain constant and only position/velocity change from
    step to step. These assumptions may be relaxed in a later version.

    TrajectoryHOOMD works a little different from the others in one respect. Because it is accessing the current
    state data directly from hoomd, there are no frames over which to loop. Accessing frame 0 will always return
    the current state of the system. Advancing forward of course must be done with hoomd run() commands.

    2D simulations will set the frame box appropriately.

    :param sysdef: System definition (returned from an init. call)
    :type: :py:meth:`hoomd.init`
    """
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

    def numParticles(self):
        """
        Get the number of particles in the trajectory.

        :return: Number of particles
        :rtype: int
        """
        return len(self.sysdef.particles);

    def __len__(self):
        """
        Get the number of frames in the trajectory, which will always be 1.

        :return: Number of frames (= 1)
        :rtype: int
        """
        return 1

    def _set_frame(self, idx):
        pass;

    def _get_current_frame(self):
        dynamic_props = {};

        # get position
        pos = numpy.array([p.position for p in self.sysdef.particles], dtype=numpy.float32);
        image = numpy.array([p.image for p in self.sysdef.particles], dtype=numpy.int32);
        velocity = numpy.array([p.velocity for p in self.sysdef.particles], dtype=numpy.float32);

        dynamic_props['position'] = pos;
        dynamic_props['image'] = image;
        dynamic_props['velocity'] = velocity;

        hoomd_box = self.sysdef.box;
        box = Box(hoomd_box.Lx, hoomd_box.Ly, hoomd_box.Lz, hoomd_box.dimensions == 2)

        return Frame(self, 0, dynamic_props, box);

class TrajectoryDISCMC(Trajectory):
    """
    Trajectory information loaded from a discmc ouptut file

    :param fname: file name to load
    :type fname: string
    """

    def __init__(self, fname):
        Trajectory.__init__(self);

        if h5py is None:
            raise RuntimeError('h5py not found')
        self.df = h5py.File(fname, 'r')

    def numParticles(self):
        """
        Get the number of particles in the trajectory.

        :return: Number of particles
        :rtype: int
        """
        return self.df["/param/N"][0];

    def __len__(self):
        """
        Get the number of frames in the trajectory.

        :return: Number of frames
        :rtype: int
        """
        return self.df["/traj/step"].shape[0];

    def _set_frame(self, idx):
        self.cur_frame = idx;

    def _get_current_frame(self):
        dynamic_props = {};

        # get position
        pos = numpy.zeros(shape=(self.numParticles(), 3), dtype=numpy.float32);

        dset_pos = self.df["/traj/pos"];
        m = self.df["/param/m"][0];
        w = self.df["/param/w"][0];
        L = m * w;

        if self.cur_frame < dset_pos.shape[0]:
            pos[:,0:2] = dset_pos[self.cur_frame,:];
            # freud boxes are centered on 0,0,0 - shift the coordinates
            pos[:,0] -= L/2.0;
            pos[:,1] -= L/2.0;
            dynamic_props['position'] = pos;

        dynamic_props['rho'] = float(self.numParticles())/(L*L)
        dynamic_props['m'] = m;
        dynamic_props['w'] = w;

        # extract M(r) data
        if "/param/dr" in self.df:
            dynamic_props['dr'] = self.df["/param/dr"][0];
            Mr = numpy.zeros(self.df["/traj/Mr"].shape[0]);
            Mr = self.df["/traj/Mr"][self.cur_frame,:];
            dynamic_props['Mr'] = Mr;

        box = Box(L, L, 0, True);

        return Frame(self, 0, dynamic_props, box);
