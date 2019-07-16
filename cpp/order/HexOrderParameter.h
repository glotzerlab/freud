// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef HEX_ORDER_PARAMETER_H
#define HEX_ORDER_PARAMETER_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "OrderParameter.h"
#include "VectorMath.h"

/*! \file HexOrderParameter.h
    \brief Compute the hexatic order parameter for each particle.
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
 */
class HexOrderParameter : public OrderParameter<unsigned int>
{
public:
    //! Constructor
    HexOrderParameter(unsigned int k = 6);

    //! Destructor
    ~HexOrderParameter();

        //! Get a reference to the last computed psi
    std::shared_ptr<std::complex<float>> getPsi()
    {
        return m_psi_array;
    }

    //! Compute the hex order parameter
    void compute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);
};

}; }; // end namespace freud::order

#endif // HEX_ORDER_PARAMETER_H
