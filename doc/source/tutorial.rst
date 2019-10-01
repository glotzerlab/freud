.. _tutorial:

========
Tutorial
========

This tutorial provides a complete introduction to **freud**.
Rather than attempting to touch on all features in **freud**, it focuses on common core concepts that will help understand how **freud** works with data and exposes computations to the user.
The tutorial begins by introducing the fundamental concepts of periodic systems as implemented in **freud** and the concept of ``Compute`` classes, which consitute the primary API for performing calculations with **freud**.
The tutorial then discusses the most common calculation performed in **freud**, finding neighboring points in periodic systems. 
The package's neighbor finding tools are tuned for high performance neighbor finding, which is what enables most of other calculations in **freud**, which typically involve characterizing local environments of points in some way.
The next part of the tutorial discusses the role of histograms in **freud**, focusing on the common features and properties that all histograms share.
Finally, the tutorial includes a few more complete demonstrations of using **freud** that should provide reasonable templates for use with almost any other features in **freud**.

.. toctree::
   :maxdepth: 2

   tutorial/periodic.rst
   tutorial/computeclass.rst
   tutorial/neighborfinding
