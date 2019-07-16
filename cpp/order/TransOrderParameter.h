// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef TRANS_ORDER_PARAMETER_H
#define TRANS_ORDER_PARAMETER_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "OrderParameter.h"
#include "VectorMath.h"

/*! \file TransOrderParameter.h
    \brief Compute the translational order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the translational order parameter for a set of points
/*!
 */
class TransOrderParameter : public OrderParameter<float>
{
public:
    //! Constructor
    TransOrderParameter(float k = 6);

    //! Destructor
    ~TransOrderParameter();

    //! Compute the translational order parameter
    void compute(const freud::locality::NeighborList* nlist,
                 const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

    //! Get a reference to the last computed dr
    std::shared_ptr<std::complex<float>> getDr()
    {
        return m_psi_array;
    }
};

}; }; // end namespace freud::order

#endif // TRANS_ORDER_PARAMETER_H
