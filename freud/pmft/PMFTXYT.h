// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXYT_H
#define PMFTXYT_H

#include "NeighborQuery.h"
#include "NeighborList.h"
#include "PMFT.h"
#include <memory>
#include "VectorMath.h"

/*! \file PMFTXYT.h
    \brief Routines for computing potential of mean force and torque in XYT coordinates
*/

namespace freud { namespace pmft {

class PMFTXYT : public PMFT
{
public:
    //! Constructor
    PMFTXYT(float x_max, float y_max, unsigned int n_x, unsigned int n_y, unsigned int n_t);

    /*! Compute the PCF for the passed in set of points. The function will be added to previous values
        of the PCF
    */
    void accumulate(const std::shared_ptr<locality::NeighborQuery>& neighbor_query, const float* orientations,
                    const vec3<float>* query_points, const float* query_orientations,
                    unsigned int n_query_points, std::shared_ptr<locality::NeighborList> nlist,
                    const freud::locality::QueryArgs& qargs);

protected:
    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduce() override;

    float m_jacobian;
};

}; }; // end namespace freud::pmft

#endif // PMFTXYT_H
