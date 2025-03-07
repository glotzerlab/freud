// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_DENSITY_H
#define LOCAL_DENSITY_H

#include <memory>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file LocalDensity.h
    \brief Routines for computing local density around a point.
*/

namespace freud { namespace density {

//! Compute the local density at each point
/*!
 */
class LocalDensity
{
public:
    //! Constructor
    LocalDensity(float r_max, float diameter);

    //! Destructor
    ~LocalDensity() = default;

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

    //! Return the cutoff distance.
    float getDiameter() const
    {
        return m_diameter;
    }

    //! Compute the local density
    void compute(const std::shared_ptr<locality::NeighborQuery>& neighbor_query,
                 const vec3<float>* query_points, unsigned int n_query_points,
                 const std::shared_ptr<locality::NeighborList>& nlist,
                 const freud::locality::QueryArgs& qargs);

    //! Get a shared pointer to the last computed density
    std::shared_ptr<const util::ManagedArray<float>> getDensity() const
    {
        return m_density_array;
    }

    //! Get a shared pointer to the last computed number of neighbors
    std::shared_ptr<const util::ManagedArray<float>> getNumNeighbors() const
    {
        return m_num_neighbors_array;
    }

private:
    box::Box m_box;   //!< Simulation box where the particles belong
    float m_r_max;    //!< Maximum neighbor distance
    float m_diameter; //!< Diameter of the particles

    std::shared_ptr<util::ManagedArray<float>> m_density_array;       //!< density array computed
    std::shared_ptr<util::ManagedArray<float>> m_num_neighbors_array; //!< number of neighbors array computed
};

}; }; // end namespace freud::density

#endif // LOCAL_DENSITY_H
