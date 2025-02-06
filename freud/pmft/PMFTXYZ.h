// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXYZ_H
#define PMFTXYZ_H

#include "NeighborList.h"
#include "NeighborQuery.h"
#include "PMFT.h"
#include "VectorMath.h"
#include <memory>

/*! \file PMFTXYZ.h
    \brief Routines for computing 3D potential of mean force in XYZ coordinates
*/

namespace freud { namespace pmft {

class PMFTXYZ : public PMFT
{
public:
    //! Constructor
    PMFTXYZ(float x_max, float y_max, float z_max, unsigned int n_x, unsigned int n_y, unsigned int n_z);

    /*! Compute the PCF for the passed in set of points. The function will be added to previous values
        of the pcf
    */
    void accumulate(const std::shared_ptr<locality::NeighborQuery>& neighbor_query,
                    const quat<float>* query_orientations, const vec3<float>* query_points,
                    unsigned int n_query_points, const quat<float>* equiv_orientations,
                    unsigned int num_equiv_orientations, std::shared_ptr<locality::NeighborList> nlist,
                    const freud::locality::QueryArgs& qargs);

    //! Reset the PMFT
    /*! Override the parent method to also reset the number of equivalent orientations.
     */
    void reset() override;

protected:
    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduce() override;

    float m_jacobian;
    unsigned int m_num_equiv_orientations; //!< The number of equivalent orientations used in the current
                                           //!< calls to compute.
};

}; }; // end namespace freud::pmft

#endif // PMFTXYZ_H
