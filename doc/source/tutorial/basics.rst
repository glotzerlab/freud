.. _basics:

======
Basics
======

Periodic Boundary Conditions
============================

The central goal of **freud** is the analysis of simulations performed in periodic boxes.
Periodic boundary conditions are ubiquitous in simulations because they permit the simulation of pseudo-infinite systems with minimal computational effort.
As long as simulation systems are sufficiently large, i.e. assuming that particles in the system experience correlations over length scales substantially smaller than the system length scale, periodic boundary conditions ensure that the system appears effectively infinite to all particles.

In order to consistently define the geometry of a simulation system with periodic boundaries, **freud** defines the :py:class:`freud.box.Box` class.
The class encapsulates the concept of a triclinic simulation box in a right-handed coordinate system.
Triclinic boxes are defined as parallelepipeds: three-dimensional polyhedra where every face is a parallelogram.
In general, any such box can be represented by three distinct, linearly independent box vectors.
Enforcing the requirement of right-handedness guarantees that the box can be represented by a matrix of the form

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{h}& =& \left(\begin{array}{ccc} L_x & xy L_y & xz L_z \\
                                            0   & L_y    & yz L_z \\
                                            0   & 0      & L_z    \\
                         \end{array}\right)
    \end{eqnarray*}

where each column is one of the box vectors.

As such, the box is characterized by six parameters: the box vector lengths :math:`L_x`, :math:`L_y`, and :math:`L_z`, and the tilt factors :math:`xy`, :math:`xz`, and :math:`yz`.
The tilt factors are directly related to the angles between the box vectors.
All computations in **freud** are built around this class, ensuring that they naturally handle data from simulations conducted in non-cubic systems.
There is also native support for two-dimensional (2D) systems when setting :math:`L_z = 0`.

Boxes can be constructed in a variety of ways.
For simple use-cases, one of the factory functions of the :py:class:`freud.box.Box` provides the easiest possible interface:

.. code-block:: python

    # Make a 10x10 square box (for 2-dimensional systems).
    freud.box.Box.square(10)

    # Make a 10x10x10 cubic box.
    freud.box.Box.cube(10)

For more complex use-cases, the :py:meth:`freud.box.Box.from_box` method provides an interface to create boxes from any object that can easily be interpreted as a box.

.. code-block:: python

    # Create a 10x10 square box from a list of two items.
    freud.box.Box.from_box([10, 10])

    # Create a 10x10x10 cubic box from a list of three items.
    freud.box.Box.from_box([10, 10, 10])

    # Create a triclinic box from a list of six items (including tilt factors).
    freud.box.Box.from_box([10, 5, 2, 0.1, 0.5, 0.7])

    # Create a triclinic box from a dictionary.
    freud.box.Box.from_box(dict(Lx=8, Ly=7, Lz=10, xy=0.5, xz=0.7, yz=0.2))

    # Directly call the constructor.
    freud.box.Box(Lx=8, Ly=7, Lz=10, xy=0.5, xz=0.7, yz=0.2, dimensions=3)


.. note::
    All **freud** boxes are centered at the origin, so for a given box the
    range of possible positions is :math:`[-L/2, L/2)`.


Compute Classes
===============

Calculations in **freud** are built around the concept of *compute classes*, Python objects that encode a given method and expose it through a ``compute`` method.
In general, these methods operate on a particular triclinic box, which can be provided as any object that can be interpreted as a **freud** box (as demonstrated in the examples above).
We can look at the :py:class:`freud.order.HexOrderParameter` class as an example:

.. code-block:: python

    import freud
    positions = ...  # Read positions from trajectory file.
    op = freud.order.HexOrderParameter(rmax=3, k=6).compute(
        (box={'Lx': 5, 'Ly': 5, 'dimensions': 2}, points=positions))

    # Plot the value of the order parameter.
    from matplotlib import pyplot as plt
    plt.plot(op.psi)

Here, we are calculating the hexatic order parameter, then using Matplotlib to plot it.
For this example, points must be provided as a list or an array of positions like :code:`[x, y, 0]` because this order parameter is designed for 2D systems.
Note that in general, the points may be provided as anything that can be interpreted as an :math:`N\times 3` list of positions, in particular either a list of lists or a NumPy array of shape :math:`(N, 3)`.
