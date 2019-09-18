// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef HEX_TRANS_ORDER_PARAMETER_H
#define HEX_TRANS_ORDER_PARAMETER_H

#include <complex>
#include <memory>
#include <tbb/tbb.h>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborComputeFunctional.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file HexTransOrderParameter.h
    \brief Compute the hexatic/translational order parameter for each particle.
*/

namespace freud { namespace order {

//! Parent class for HexOrderParameter and TransOrderParameter
/*!
 */
template<typename T> class HexTransOrderParameter
{
public:
    //! Constructor
    HexTransOrderParameter(T k): m_k(k) {}

    //! Destructor
    virtual ~HexTransOrderParameter() {}

    T getK() const
    {
        return m_k;
    }

    //! Get a reference to the order parameter array
    const util::ManagedArray<std::complex<float>> &getOrder() const
    {
        return m_psi_array;
    }


protected:
    //! Compute the order parameter
    template<typename Func>
    void computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                        const freud::locality::NeighborQuery* points,
                        freud::locality::QueryArgs qargs);

    const T m_k;
    util::ManagedArray<std::complex<float>> m_psi_array; //!< psi array computed
};

//! Compute the translational order parameter for a set of points
/*!
 */
class TransOrderParameter : public HexTransOrderParameter<float>
{
public:
    //! Constructor
    TransOrderParameter(float k = 6);

    //! Destructor
    ~TransOrderParameter();

    //! Compute the translational order parameter
    void compute(const freud::locality::NeighborList* nlist,
                 const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);
};

//! Compute the hexatic order parameter for a set of points
/*!
 */
class HexOrderParameter : public HexTransOrderParameter<unsigned int>
{
public:
    //! Constructor
    HexOrderParameter(unsigned int k = 6);

    //! Destructor
    ~HexOrderParameter();

    //! Compute the hexatic order parameter
    void compute(const freud::locality::NeighborList* nlist,
                 const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);
};

}; }; // end namespace freud::order

#endif // HEX_TRANS_ORDER_PARAMETER_H
