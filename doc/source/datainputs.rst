.. _datainputs:

=====================================
Reading Simulation Data for **freud**
=====================================

The **freud** package is designed for maximum flexibility by making minimal assumptions about its data.
However, users accustomed to the more restrictive patterns of most other tools may find this flexibility confusing.
In particular, knowing how to provide data from specific simulation sources can be a significant source of confusion.
This page is intended to describe how various types of data may be converted into a form suitable for **freud**.

To simplify the examples below, we will assume in all cases that the user wishes to compute a :class:`radial distribution function <freud.density.RDF>` over all frames in the trajectory and that the following code has already been run:

.. code-block:: python

    import freud
    rdf = freud.density.RDF(bins=50, r_max=5)


GSD Trajectories
================

Using the GSD Python API, GSD files can be very easily integrated with **freud** as shown in :ref:`gettingstarted`. This format is natively supported by `HOOMD-blue <https://hoomd-blue.readthedocs.io/>`_ and is recommended for use with freud.

.. code-block:: python

    import gsd.hoomd
    traj = gsd.hoomd.open('trajectory.gsd', 'rb')

    for frame in traj:
        rdf.compute((frame.configuration.box, frame.particles.position), reset=False)


Reading Text Files (XYZ files)
==============================

Text files, such as those in the XYZ format, are common simulation data outputs.
XYZ files can be generated and used by many tools for particle simulation and analysis, including LAMMPS and VMD.
As a result, while they are extremely easy to parse, they are also typically lacking in information.
In particular, they usually contain no information about the system box, so this must already be known by the user.
Assuming knowledge of the box used in the simulation, a LAMMPS XYZ file could be used as follows:

.. code-block:: python

    N = int(np.genfromtxt('trajectory.xyz', max_rows=1))
    traj = np.genfromtxt(
        'trajectory.xyz', skip_header=2,
        invalid_raise=False)[:, 1:4].reshape(-1, N, 3)

    for frame in traj[frame_start:]:
        rdf.compute((frame.configuration.box, frame.particles.position), reset=False)

Note that various readers do exist for XYZ files, but due to their simplicity we choose to read them in manually in this example.
The first line is the number of particles, so we read this line and use it to determine how to reshape the contents of the rest of the file into a NumPy array.

Using External Readers (MDAnalysis for DCD files)
=================================================

For many trajectory formats, high-quality readers already exist.
Packages such as `MDAnalysis <https://www.mdanalysis.org/>`_, `MDTraj <http://mdtraj.org/>`_, `pytraj <https://amber-md.github.io/pytraj/latest/read_and_write.html>`_, or `garnett <https://garnett.readthedocs.io/>`_ can read virtually all common trajectory formats.

DCD files are among the most familiar simulation outputs due to their longevity.
Here, we provide an example using `MDAnalysis <https://www.mdanalysis.org/>`_ to read data from a DCD file.

.. code-block:: python

    import MDAnalysis
    reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')

    for frame in reader:
        rdf.compute((
            freud.box.Box.from_matrix(frame.triclinic_dimensions),
            frame.positions), reset=False)
