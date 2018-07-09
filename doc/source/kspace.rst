=============
KSpace Module
=============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.kspace.FTbase
    freud.kspace.FTdelta
    freud.kspace.FTsphere
    freud.kspace.FTpolyhedron
    freud.kspace.SFactor3DPoints
    freud.kspace.AnalyzeSFactor3D
    freud.kspace.SingleCell3D
    freud.kspace.FTfactory
    freud.kspace.FTconvexPolyhedron
    freud.kspace.Constraint
    freud.kspace.AlignedBoxConstraint
    freud.kspace.DeltaSpot
    freud.kspace.GaussianSpot
    freud.kspace.InterpolatedDeltaSpot
    freud.kspace.meshgrid2
    freud.kspace.rotate
    freud.kspace.quatrot
    freud.kspace.constrainedLatticePoints
    freud.kspace.reciprocalLattice3D

.. rubric:: Details

.. automodule:: freud.kspace
    :synopsis: Compute various quantities in reciprocal space.

Structure Factor
================

.. autoclass:: freud.kspace.SFactor3DPoints(box, g)
    :members:

.. autoclass:: freud.kspace.AnalyzeSFactor3D(S)
    :members:

.. autoclass:: freud.kspace.SingleCell3D(k, ndiv, dK, boxMatrix)
    :members:

.. autoclass:: freud.kspace.FTfactory()
    :members:

.. autoclass:: freud.kspace.FTbase()
    :members:

.. autoclass:: freud.kspace.FTdelta()
    :members:

.. autoclass:: freud.kspace.FTsphere()
    :members:

.. autoclass:: freud.kspace.FTpolyhedron()
    :members:

.. autoclass:: freud.kspace.FTconvexPolyhedron()
    :members:

Diffraction Patterns
====================

.. autoclass:: freud.kspace.DeltaSpot()
    :members:

.. autoclass:: freud.kspace.GaussianSpot()
    :members:

Utilities
=========

.. autoclass:: freud.kspace.Constraint()
    :members:

.. autoclass:: freud.kspace.AlignedBoxConstraint()
    :members:

.. autofunction:: freud.kspace.constrainedLatticePoints()

.. autofunction:: freud.kspace.reciprocalLattice3D()

.. autofunction:: freud.kspace.meshgrid2(*arrs)
