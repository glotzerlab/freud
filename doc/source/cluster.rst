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
    :synopsis: Find clusters

Cluster
=======

.. autoclass:: freud.cluster.Cluster(box, rcut)
    :members: computeClusterMembership, computeClusters

Cluster Properties
==================

.. autoclass:: freud.cluster.ClusterProperties(box)
    :members: computeProperties
