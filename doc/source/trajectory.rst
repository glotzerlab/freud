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

.. autoclass:: freud.trajectory.Box(*args, **kwargs)
    :members:

Trajectory Frame
================

.. autoclass:: freud.trajectory.Frame(traj, idx, dynamic_props, box, timestep)
    :members:

Trajectory Loaders
==================

.. autoclass:: freud.trajectory.Trajectory
    :members:

.. autoclass:: freud.trajectory.TrajectoryVMD(mol_id=None)
    :members:

.. autoclass:: freud.trajectory.TrajectoryXML(xml_fname_list, dynamic)
    :members:

.. autoclass:: freud.trajectory.TrajectoryXMLDCD(xml_fname, dcd_fname)
    :members:

.. autoclass:: freud.trajectory.TrajectoryPOS(pos_fname, dynamic)
    :members:

.. autoclass:: freud.trajectory.TrajectoryHOOMD(sysdef)
    :members:

.. autoclass:: freud.trajectory.TrajectoryDISCMC(fname)
    :members:
