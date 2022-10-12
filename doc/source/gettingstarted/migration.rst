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
    * - Color voronoi diagram by area of the cells.
      - ``N/A``
      - ``voro.plot(..., color_by='area')``
