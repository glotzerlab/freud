// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_DENSITY_H
#define LOCAL_DENSITY_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NeighborList.h"

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
        LocalDensity(float r_cut, float volume, float diameter);

       //! Destructor
       ~LocalDensity();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the local density
        void compute(const box::Box &box,
                     const freud::locality::NeighborList *nlist,
                     const vec3<float> *ref_points,
                     unsigned int n_ref,
                     const vec3<float> *points,
                     unsigned int Np);

        //! Get the number of reference particles
        unsigned int getNRef();

        //! Get a reference to the last computed density
        std::shared_ptr< float > getDensity();

        //! Get a reference to the last computed number of neighbors
        std::shared_ptr< float > getNumNeighbors();

    private:
        box::Box m_box;                   //!< Simulation box where the particles belong
        float m_rcut;                     //!< Maximum neighbor distance
        float m_volume;                   //!< Volume (area in 2d) of a single particle
        float m_diameter;                 //!< Diameter of the particles
        unsigned int m_n_ref;             //!< Last number of points computed

        std::shared_ptr< float > m_density_array;         //!< density array computed
        std::shared_ptr< float > m_num_neighbors_array;   //!< number of neighbors array computed
    };

}; }; // end namespace freud::density

#endif // LOCAL_DENSITY_H
