.. _computeclass:

===============
Compute Classes
===============

Calculations in **freud** are built around the concept of ``Compute classes``, Python objects that encode a given method and expose it through a ``compute`` method.
In general, these methods operate on a particular triclinic box, which can be provided as any object that can be interpreted as a **freud** box (as demonstrated in the examples above).
We can look at the :class:`freud.order.Hexatic` order parameter calculator as an example:

.. code-block:: python

    import freud
    positions = ...  # Read positions from trajectory file.
    op = freud.order.Hexatic(k=6)
    op.compute(
        ({'Lx': 5, 'Ly': 5, 'dimensions': 2}, positions),
        dict(r_max=3)
    )

    # Plot the value of the order parameter.
    from matplotlib import pyplot as plt
    plt.hist(np.absolute(op.order))

Here, we are calculating the hexatic order parameter, then using Matplotlib to plot 
The :class:`freud.order.Hexatic` class constructor accepts a single argument :code:`k`, which represents the periodicity of the calculation.
If you're unfamiliar with this order parameter, the most important piece of information here is that many compute methods in **freud** require parameters that are provided when the ``Compute class`` is constructed.

To calculate the order parameter we call :meth:`compute <freud.order.Hexatic.compute>`, which takes two arguments, a :class:`tuple` `(box, points)` and a :class:`dict`.
We first focus on the first argument.
The ``box`` is any object that can be coerced into a :class:`freud.box.Box` as described in the previous section; in this case, we use a dictionary to specify a square (2-dimensional) box.
The ``points`` must be anything that can be coerced into a 2-dimensional NumPy array of shape :code:`(N, 3)`
In general, the points may be provided as anything that can be interpreted as an :math:`N\times 3` list of positions; for more details on valid inputs here, see :func:`numpy.asarray`.
Note that because the hexatic order parameter is designed for two-dimensional systems, the points must be provided of the form `[x, y, 0]` (i.e. the z-component must be 0).
We'll go into more detail about the ``(box, points)`` tuple `soon <paircompute>`_, but for now, it's sufficient to just think of it as specifying the system of points we want to work with.

Now let's return to the second argument to ``compute``, which is a dictionary is used to determine which particle neighbors to use.
Many computations in **freud** (such as the hexatic order parameter) involve the bonds in the system (for example, the average length of bonds or the average number of bonds a given point has).
However, the concept of a bond is sufficiently variable between different calculations; for instance, should points be considered bonded if they are within a certain distance of each other?
Should every point be considered bonded to a fixed number of other points?

To accommodate this variability, **freud** offers a very general framework by which bonds can be specified, and we'll go into more details in the `next section <neighborfinding>`_.
In the example above, we've simply informed the :class:`Hexatic <freud.order.Hexatic>` class that we want it to define bonds as pairs of particles that are less than 3 distance units apart.
We then access the computed order parameter as ``op.order`` (we use :func:`np.absolute` because the output is a complex number and we just want to see its magnitude).


Accessing Computed Properties
=============================

In general, compute classes expose their calculations using `properties <https://docs.python.org/3/library/functions.html#property>`_.
Any parameters to the ``Compute`` object (e.g. :code:`k` in the above example) can typically be accessed as soon as the object is constructed:

.. code-block:: python

    op = freud.order.Hexatic(k=6)
    op.k

Computed quantities can also be accessed in a similar manner, but only once the ``compute`` method is called.
For example:

.. code-block:: python

    op = freud.order.Hexatic(k=6)

    # This will raise an exception
    op.order

    op.compute(
        ({'Lx': 5, 'Ly': 5, 'dimensions': 2}, positions),
        dict(r_max=3)
    )

    # Now you can access this.
    op.order

.. note::
    Most (but not all) of **freud**'s ``Compute classes`` are Python wrappers
    around high-performance implementations in C++. As a result, none of the
    data or the computations is actually stored in the Python object. Instead,
    the Python object just stores an instance of the C++ object that actually
    owns all its data, performs calculations, and returns computed quantities
    to the user. Python properties provide a nice way to hide this logic so
    that the Python code involves just a few lines.

Compute objects is that they can be used many times to calculate quantities, and the most recently calculated output can be accessed through the property.
If you need to perform a series of calculations and save all the data, you can also easily do that:

.. code-block:: python
    
    # Recall that lists of length 2 automatically convert to 2d freud boxes.
    box = [5, 5]

    op = freud.order.Hexatic(k=6)

    # Assuming that we have a list of Nx3 NumPy arrays that represents a
    # simulation trajectory, we can loop over it and calculate the order
    # parameter values in sequence.
    trajectory  = ...  # Read trajectory file into a list of positions by frame.
    hexatic_values = []
    for points in trajectory:
        op.compute((box, points), dict(r_max=3))
        hexatic_values.append(op.order)


To make using **freud** as simple as possible, all ``Compute classes`` are designed to return ``self`` when compute is called.
This feature enables a very concise *method-chaining* idiom in **freud** where computed properties are accessed immediately:

.. code-block:: python

    order = freud.order.Hexatic(k=6).compute(
        (box, points)).order

