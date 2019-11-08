.. _specialtopics:

==============
Special Topics
==============

Some of the most central components have a high level of abstraction.
This abstraction has multiple advantages: it dramatically simplifies the process of implementing new code, it reduces duplication throughout the code base, and ensures that bug fixes and optimization can occur along a single path for the entire module.
However, this abstraction comes at the cost of significant complexity.
This documentation should help orient developers familiarizing themselves with these topics by providing high-level overviews of how these parts of the code are structured and how the pieces fit together.

.. toctree::
   :maxdepth: 2

   specialtopics/memory
   specialtopics/neighbors
   specialtopics/histograms
