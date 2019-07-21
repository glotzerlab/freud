================
Getting Started
================

Once you have freud `installed <installation.rst>`_, you can start using **freud** with any simulation data that you have on hand.
As an example, we'll assume that you have run a simulation using the `HOOMD-blue <http://glotzerlab.engin.umich.edu/hoomd-blue/>`_ and used the ``hoomd.dump.gsd`` command to output the trajectory into a file ``trajectory.gsd``
The GSD file format provides its own convenient Python file reader that offers access to data in the form of NumPy arrays, making it immediately suitable for calculation with **freud**.

We start by reading the data into a NumPy array:

.. code-block:: python

    import gsd.hoomd, gsd.pygsd

    f = gsd.pygsd.GSDFile(open('trajectory.gsd', 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)


We can now immediately calculate important quantities.
Here, we will compute the radial distribution function :math:`g(r)` using the `freud.density.RDF` compute class.
Since the radial distribution function is in practice computed as a histogram, we must specify the histogram bin widths and the largest interparticle distance to include in our calculation.
To do so, we simply instantiate the class with the appropriate parameters and then pass in the data, taking advantage of **freud**'s *method chaining* API to do both at once:

.. code-block:: python

    import freud
    rdf = freud.density.RDF(rmax=5, dr=0.1).compute(traj[-1].configuration.box, traj[-1].particles.position)

We can now access the data through properties of the ``rdf`` object; for example, we might plot the data using Matplotlib:

.. code-block:: python

    import matplotlib as plt
    fig, ax = plt.subplots()
    ax.plot(rdf.R, rdf.RDF)

You will note that in the above example, we computed `g(r)` only using the final frame of the simulation trajectory.
However, in many cases, radial distributions and other similar quantites may be noisy in simulations due to the natural fluctuations present.
In general, what we are interested in are *time-averaged* quantities once a system has equilibrated.
To perform such a calculation, we can easily modify our original calculation to take advantage of **freud**'s *accumulation* features.
Assuming that you have some method for identifying the frames you wish to include in your sample, our original code snippet would be modified as follows:


.. code-block:: python

    import freud
    rdf = freud.density.RDF(rmax=5, dr=0.1)
    for frame in traj[frame_start:]:
        rdf.accumulate(frame.configuration.box, frame.particles.position)

You can then access the data exactly as we previously did.

And that's it!
You now know enough to start making use of **freud**.
If you'd like a complete walkthrough please look at the `tutorial`_.
To see a more thorough listing of the features in **freud** and how they can be used, look through the `examples`_ or parse the `API documentation <modules>`_ for the module you're interested in.
