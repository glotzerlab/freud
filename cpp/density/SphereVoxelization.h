// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SPHERE_VOXELIZATION_H
#define SPHERE_VOXELIZATION_H

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "ThreadStorage.h"
#include "VectorMath.h"

/*! \file SphereVoxelization.h
    \brief Routines for computing voxelized densities from spheres centered at points.
*/

namespace freud { namespace density {

//! Computes the voxels of a system on a grid.
/*! Replaces particle positions with a sphere and calculates the
        contribution from the grid based upon the distance of the grid cell
        from the center of the sphere.
*/
class SphereVoxelization
{
public:
    //! Constructor
    SphereVoxelization(vec3<unsigned int> width, float r_max);

    // Destructor
    ~SphereVoxelization() {}

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Return the cutoff distance.
    float getRMax() const
    {
        return m_r_max;
    }

    //! Compute the voxelization
    void compute(const freud::locality::NeighborQuery* nq);

    //! Get a reference to the last computed voxels
    const util::ManagedArray<unsigned int>& getVoxels() const;

    vec3<unsigned int> getWidth();

private:
    box::Box m_box;             //!< Simulation box where the particles belong
    vec3<unsigned int> m_width; //!< Num of bins on each side of the cube
    float m_r_max;              //!< Max r at which to compute voxels

    util::ManagedArray<unsigned int> m_voxels_array; //! computed voxels array
};

}; }; // end namespace freud::density

#endif // SPHERE_VOXELIZATION_H
