.. contents:: Freud trajectory

=================
Trajectory Module
=================

Contains data structures for simulation boxes, as well as providing methods and data structures for
reading MD/MC trajectories into numpy arrays for use in Freud modules. It is not required to use a
Freud trajectory reader to load in your data; any data appropriately formatted as a numpy array is
suitable for use in Freud. These modules are provided for convenience. Write/suggest your own for
inclusion as needed.


Simulation Box
==============

.. autoclass:: freud.trajectory.Box
    :members:

    .. method:: __init__(self)

    null constructor

    .. method:: __init__(self, L)

    initializes cubic box of side length L

    .. method:: __init__(self, L, is2D)

    initializes box of side length L (will create a 2D/3D box based on is2D)

    .. method:: __init__(self, L=L, is2D=False)

    initializes box of side length L (will create a 2D/3D box based on is2D)

    .. method:: __init__(self, Lx, Ly, Lz)

    initializes cubic box of side length L

    .. method:: __init__(self, Lx, Ly, Lz, is2D=False)

    initializes box with side lengths Lx, Ly (, Lz if is2D=False)

    .. method:: __init__(self, Lx=0.0, Ly=0.0, Lz=0.0, xy=0.0, xz=0.0, yz=0.0, is2D=False)

    Preferred method to initialize. Pass in as kwargs. Any not set will be set to the listed defaults.

Trajectory Frame
================

.. autoclass:: freud.trajectory.Frame
    :members:

    .. method:: __init__(self, traj, idx, dynamic_props, box, timestep)

    Initialize a frame for access. High level classes should not construct Frame classes directly. Instead create a Trajectory and query it to get frames.

    :param: traj: Parent Trajectory
    :param: idx: Index of the frame
    :param: dynamic_props: Dictionary of dynamic properties accessible in this frame
    :param: box: the simulation Box for this frame

Trajectory Loaders
==================

.. autoclass:: freud.trajectory.Trajectory
    :members:

.. autoclass:: freud.trajectory.TrajectoryVMD
    :members:

    .. method:: __init__(self, mol_id)

    Initialize a VMD trajectory for access. When a mol_id is set to None, the 'top' molecule is accessed.

    :param: mol_id: ID number of the vmd molecule to access

.. autoclass:: freud.trajectory.TrajectoryXML
    :members:

    .. method:: __init__(self, xml_fname_list, dynamic)

        Initialize a list of XMLs trajectory for access.

        :param: xml_fname_list: File names of the XML files to be read
        :param: dynamic: List of dynamic properties in the trajectory

.. autoclass:: freud.trajectory.TrajectoryXMLDCD
    :members:

    .. method:: __init__(self, xml_fname, dcd_fname)

        Initialize a list of XMLs trajectory for access.

        :param: xml_fname: File name of the XML file to read the structure from
        :param: dcd_fname: File name of the DCD trajectory to read (or None to skip reading the trajectory data)

.. autoclass:: freud.trajectory.TrajectoryPOS
    :members:

    .. method:: __init__(self, pos_fname, dynamic)

        Initialize an POS trajectory for access.

        :param: pos_fname: File name of the POS file to read the structure from
        :param: dynamic: List of dynamic properties in the trajectory

.. autoclass:: freud.trajectory.TrajectoryHOOMD
    :members:

    .. method:: __init__(self, sysdef)

        Initialize a HOOMD trajectory.

        :param: sysdef: System definition (returned from an init. call)

.. autoclass:: freud.trajectory.TrajectoryDISCMC
    :members:

    .. method:: __init__(self, fname)

        Initialize a DISCMC trajectory.

        :param: fname: file name to load
