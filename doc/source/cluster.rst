==============
Cluster Module
==============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.cluster.Cluster
    freud.cluster.ClusterProperties

.. rubric:: Details

.. automodule:: freud.cluster
    :synopsis: Find clusters of points.

Cluster
=======

.. autoclass:: freud.cluster.Cluster(box, rcut)
    :members: computeClusterMembership, computeClusters, plot

Cluster Properties
==================

.. autoclass:: freud.cluster.ClusterProperties(box)
    :members: computeProperties
