.. _quickstart:

================
Quickstart Guide
================

Once you have `installed freud <installation.rst>`_, you can start using **freud** with any simulation data that you have on hand.
As an example, we'll assume that you have run a simulation using the `HOOMD-blue <https://glotzerlab.engin.umich.edu/hoomd-blue/>`_ and used the :class:`hoomd.dump.gsd` command to output the trajectory into a file ``trajectory.gsd``.
The `GSD file format <https://gsd.readthedocs.io/en/stable/>`_ provides its own convenient Python file reader that offers access to data in the form of NumPy arrays, making it immediately suitable for calculation with **freud**. Many other file readers and data formats are supported, see :ref:`datainputs` for a full list and more examples.

We start by reading the data into a NumPy array:

.. code-block:: python

    import gsd.hoomd
    traj = gsd.hoomd.open('trajectory.gsd', 'rb')


We can now immediately calculate important quantities.
Here, we will compute the radial distribution function :math:`g(r)` using the :class:`freud.density.RDF` compute class.
Since the radial distribution function is in practice computed as a histogram, we must specify the histogram bin widths and the largest interparticle distance to include in our calculation.
To do so, we simply instantiate the class with the appropriate parameters and then perform a computation on the given data:

.. code-block:: python

    import freud
    rdf = freud.density.RDF(bins=50, r_max=5)
    rdf.compute(system=traj[-1])

We can now access the data through properties of the ``rdf`` object.

.. code-block:: python

    r = rdf.bin_centers
    y = rdf.rdf

Many classes in **freud** natively support plotting their data using `Matplotlib <https://matplotlib.org/>`:

.. code-block:: python

    import matplotlib as plt
    fig, ax = plt.subplots()
    rdf.plot(ax=ax)

You will note that in the above example, we computed :math:`g(r)` only using the final frame of the simulation trajectory, ``traj[-1]``.
However, in many cases, radial distributions and other similar quantities may be noisy in simulations due to the natural fluctuations present.
In general, what we are interested in are *time-averaged* quantities once a system has equilibrated.
To perform such a calculation, we can easily modify our original calculation to take advantage of **freud**'s *accumulation* features.
To accumulate, just add the argument ``reset=False`` with a supported compute object (such as histogram-like computations).
Assuming that you have some method for identifying the frames you wish to include in your sample, our original code snippet would be modified as follows:

.. code-block:: python

    import freud
    rdf = freud.density.RDF(bins=50, r_max=5)
    for frame in traj:
        rdf.compute(frame, reset=False)

You can then access the data exactly as we previously did.
And that's it!

Now that you've seen a brief example of reading data and computing a radial distribution function, you're ready to learn more.
If you'd like a complete walkthrough please see the :ref:`tutorial`.
The tutorial walks through many of the core concepts in **freud** in greater detail, starting with the basics of the simulation systems we analyze and describing the details of the neighbor finding logic in **freud**.
To see specific features of **freud** in action, look through the :ref:`examples`.
More detailed documentation on specific classes and functions can be found in the `API documentation <modules>`_.
