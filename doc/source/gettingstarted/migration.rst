.. _migration:

============================
Migration to freud Version 3
============================

Version 3 of the freud library introduces a few breaking changes to make the API
more intuitive and facilitate future development. This guide explains how to
alter scripts which use freud v2 APIs so they can be used with freud v3.

Overview of API Changes
=======================

.. list-table::
    :header-rows: 1

    * - Goal
      - v2 API
      - v3 API
    * - Use default RDF normalization.
      - ``freud.density.RDF(..., normalize=False)``
      - ``freud.density.RDF(..., normalization_mode='exact')``
    * - Normalize small system RDFs to 1.
      - ``freud.density.RDF(..., normalize=True)``
      - ``freud.density.RDF(..., normalization_mode='finite_size')``
    * - Color voronoi diagram by number of cell sides.
      - ``voro.plot(..., color_by_sides=True)``
      - ``voro.plot(..., color_by='sides')``
    * - Get vectors corresponding to neighbor pairs.
      - ``points[nlist.point_indices] - points[nlist.query_point_indices]``
      - ``nlist.vectors``
    * - Create a custom neighborlist from arrays.
      - ``freud.locality.NeighborList.from_arrays(..., distances=...)``
      - ``freud.locality.NeighborList.from_arrays(..., vectors=...)``
    * - Nematic API change: Nematic constructor no longer requires "nematic director" (molecular director) and compute method uses vector orientations (3D vectors) instead of quaternions (4D vectors).
      - ``freud.order.Nematic(u=).compute(orientations = 4D_quaternions)``
      - ``freud.order.Nematic().compute(orientations = 3D_orientation_vectors)``
