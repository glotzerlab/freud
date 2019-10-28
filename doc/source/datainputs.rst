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


Native Integrations
===================

The **freud** library offers interoperability with several popular tools for particle simulations, analysis, and visualization.
Below is a list of file formats and tools that are directly supported as "system-like" objects (see :class:`freud.locality.NeighborQuery.from_system`).
Such "system-like" objects are data containers that store information about a periodic box and particle positions.
Other attributes, such as particle orientations, are not included automatically in the system representation and must be loaded as separate NumPy arrays.

GSD Trajectories
----------------

Using the GSD Python API, GSD files can be easily integrated with **freud** as shown in :ref:`gettingstarted`.
This format is natively supported by `HOOMD-blue <https://hoomd-blue.readthedocs.io/>`_ and is recommended for use with freud because it also stores other quantities, such as orientations, that are required by some analysis methods in **freud**.
Here, we provide an example that reads data from a GSD file.

.. code-block:: python

    import gsd.hoomd
    traj = gsd.hoomd.open('trajectory.gsd', 'rb')

    for frame in traj:
        rdf.compute(frame, reset=False)

MDAnalysis Readers
------------------

The `MDAnalysis <https://www.mdanalysis.org/>`_ package can read `many popular trajectory formats <https://www.mdanalysis.org/docs/documentation_pages/coordinates/init.html#supported-coordinate-formats>`_, including common output formats from CHARMM, NAMD, LAMMPS, Gromacs, Tinker, AMBER, GAMESS, and more.

DCD files are among the most familiar simulation outputs due to their longevity.
Here, we provide an example that reads data from a DCD file.

.. code-block:: python

    import MDAnalysis
    reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')

    for frame in reader:
        rdf.compute(frame, reset=False)

garnett Trajectories
--------------------

The `garnett <https://garnett.readthedocs.io/>`_ package can read `several trajectory formats <https://garnett.readthedocs.io/en/stable/readerswriters.html#file-formats>`_ that have historically been supported by the HOOMD-blue simulation engine, as well as other common types such as DCD and CIF.
The **garnett** package will auto-detect supported file formats by the file extension.
Here, we provide an example that reads data from a POS file.

.. code-block:: python

    import garnett

    with garnett.read('trajectory.pos') as traj:
        for frame in traj:
            rdf.compute(frame, reset=False)

OVITO Modifiers
---------------

The `OVITO Open Visualization Tool <https://www.ovito.org/>`_ supports user-written Python modifiers.
The **freud** package can be installed alongside OVITO to enable user modifiers that leverage analyses from **freud**.
Below is an example modifier that creates a user particle property in the OVITO pipeline for Steinhardt bond order parameters.

.. code-block:: python

    import freud

    def modify(frame, data):
        ql = freud.order.Steinhardt(l=6)
        ql.compute(data, neighbors={'num_neighbors': 6})
        data.create_user_particle_property(
            name='ql', data_type=float, data=ql.ql)
        print('Created ql property for {} particles.'.format(data.particles.count))

HOOMD-blue Snapshots
--------------------

`HOOMD-blue <https://hoomd-blue.readthedocs.io/en/stable/>`_ supports analyzers, callback functions that can perform analysis.
Below is an example demonstrating how to use an anlyzer to log the Steinhardt bond order parameter :math:`q_6` during the simulation run.

.. code-block:: python

    import hoomd
    from hoomd import md
    import freud

    hoomd.context.initialize()

    # Create a 10x10x10 simple cubic lattice of particles with type name A
    system = hoomd.init.create_lattice(
        unitcell=hoomd.lattice.sc(a=2.0, type_name='A'), n=10)

    # Specify Lennard-Jones interactions between particle pairs
    nl = md.nlist.cell()
    lj = md.pair.lj(r_cut=3.0, nlist=nl)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

    # Integrate at constant temperature
    md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.2, seed=4)

    # Create a Steinhardt object to analyze snapshots
    ql = freud.order.Steinhardt(l=6)

    def compute_q6(timestep):
        print(timestep)
        snap = system.take_snapshot()
        ql.compute(snap, neighbors={'num_neighbors': 6})
        return ql.order

    # Register a logger that computes q6 and saves to a file
    ql_logger = hoomd.analyze.log(filename='ql.dat', quantities=['q6'], period=100)
    ql_logger.register_callback('q6', compute_q6)

    # Run for 10,000 time steps
    hoomd.run(10e3)

Reading Text Files (XYZ files)
==============================

Typically, it is best to use one of the natively supported data readers described above, however it is sometimes necessary to parse trajectory information directly from a text file.
XYZ files can be generated and used by many tools for particle simulation and analysis, including LAMMPS and VMD.
Note that various readers do exist for XYZ files, include MDAnalysis, but we choose to read the file manually for this example.
Though they are extremely easy to parse, XYZ files usually contain no information about the system box, so this must already be known by the user.
Assuming knowledge of the box used in the simulation, a LAMMPS XYZ file could be used as follows:

.. code-block:: python

    N = int(np.genfromtxt('trajectory.xyz', max_rows=1))
    traj = np.genfromtxt(
        'trajectory.xyz', skip_header=2,
        invalid_raise=False)[:, 1:4].reshape(-1, N, 3)
    box = freud.Box.cube(L=20)

    for frame_positions in traj:
        rdf.compute((box, frame_positions), reset=False)

The first line is the number of particles, so we read this line and use it to determine how to reshape the contents of the rest of the file into a NumPy array.

Other External Readers (MDAnalysis for DCD files)
=================================================

For many trajectory formats, high-quality readers already exist.
DCD files are among the most familiar simulation outputs due to their longevity.
Here, we provide an example using `MDAnalysis <https://www.mdanalysis.org/>`_ to read data from a DCD file.

.. code-block:: python

    import MDAnalysis
    reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')

    for frame in reader:
        rdf.compute((
            freud.box.Box.from_matrix(frame.triclinic_dimensions),
            frame.positions), reset=False)
