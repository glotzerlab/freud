.. _datainputs:

=====================================
Reading Simulation Data for **freud**
=====================================

The **freud** package is designed for maximum flexibility by making minimal assumptions about its data.
However, users accustomed to the more restrictive patterns of most other tools may find this flexibility confusing.
In particular, knowing how to provide data from specific simulation sources can be a significant source of confusion.
This page is intended to describe how various types of data may be converted into a form suitable for **freud**

To simplify the examples below, we will assume in all cases that the user wishes to compute a :class:`radial distribution function <freud.density.RDF>` and that the following code has already been run:

.. code-block:: python

    import freud
    rdf = freud.density.RDF(bins=50, rmax=5)


GSD Trajectories
================

Using the GSD Python API, GSD files can be very easily integrated with **freud** as shown in :ref:`gettingstarted`

.. code-block:: python

    import gsd.hoomd
    traj = gsd.hoomd.open('trajectory.gsd', 'rb')

    for frame in traj:
        rdf.accumulate((frame.configuration.box, frame.particles.position))


XYZ Files
=========

XYZ files are among the simplest data outputs.
As a result, while they are extremely easy to parse, they are also typically lacking in information.
In particular, they usually contain no information about the system box, so this must already be known by the user.
Assuming knowledge of the box used in the simulation, a LAMMPS XYZ file could be used as follows:

.. code-block:: python

	N = int(np.genfromtxt('trajectory.xyz', max_rows=1))
	traj = np.genfromtxt('trajectory.xyz', skip_header=2,
		invalid_raise=False)[:, 1:4].reshape(-1, N, 3

    for frame in traj[frame_start:]:
        rdf.accumulate((frame.configuration.box, frame.particles.position))

Note that various readers do exist for XYZ files, but due to their simplicity we simply choose to read them in manually in this example.
The first line is the number of particles, so we simply read this then use it to determine how to reshape the contents of the rest of the file into a NumPy array.

DCD Files
=========

DCD files are among the most familiar simulation outputs due to their longevity.
As a result, numerous high-quality DCD readers also already exist.
Here, we provide an example using `MDAnalysis <https://www.mdanalysis.org/>`_ to read the data, but we could just as easily make use of another reader such as `MDTraj <http://mdtraj.org/1.6.2/api/generated/mdtraj.load_dcd.html#mdtraj.load_dcd>`_ or `pytraj <https://amber-md.github.io/pytraj/latest/read_and_write.html>`_.

.. code-block:: python

    reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')
    for frame in reader:
        rdf.accumulate((
            freud.box.Box.from_matrix(frame.triclinic_dimensions),
            frame.positions))
