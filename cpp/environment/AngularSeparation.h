// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ANGULAR_SEPARATION_H
#define ANGULAR_SEPARATION_H

#include <tbb/tbb.h>

#include "NeighborList.h"
#include "VectorMath.h"

/*! \file AngularSeparation.h
    \brief Compute the angular separation for each particle.
*/

namespace freud { namespace environment {

float computeSeparationAngle(const quat<float> ref_q, const quat<float> q);

float computeMinSeparationAngle(const quat<float> ref_q, const quat<float> q, const quat<float>* equiv_qs,
                                unsigned int Nequiv);

//! Compute the angular separation for a set of points
/*!
 */
class AngularSeparation
{
public:
    //! Constructor
    AngularSeparation();

    //! Destructor
    ~AngularSeparation();

    //! Compute the angular separation between neighbors
    void computeNeighbor(const quat<float>* orientations, unsigned int n_points,
                         const quat<float>* query_orientations, unsigned int n_query_points,
                         const quat<float>* equiv_orientations, unsigned int n_equiv_orientations,
                         const freud::locality::NeighborList* nlist);

    //! Compute the angular separation with respect to global orientation
    void computeGlobal(const quat<float>* global_orientations, unsigned int n_global,
                       const quat<float>* orientations, unsigned int n_points,
                       const quat<float>* equiv_orientations, unsigned int n_equiv_orientations);

    //! Returns the last computed neighbor angle array
    util::ManagedArray<float> &getNeighborAngles()
    {
        return m_neighbor_angles;
    }

    //! Returns the last computed global angle array
    util::ManagedArray<float> &getGlobalAngles()
    {
        return m_global_angles;
    }

private:
    util::ManagedArray<float> m_neighbor_angles;  //!< neighbor angle array computed
    util::ManagedArray<float> m_global_angles; //!< global angle array computed
};

}; }; // end namespace freud::environment

#endif // ANGULAR_SEPARATION_H
