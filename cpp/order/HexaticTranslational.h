// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef HEXATIC_TRANSLATIONAL_H
#define HEXATIC_TRANSLATIONAL_H

#include <complex>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborComputeFunctional.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file HexaticTranslational.h
    \brief Compute the hexatic/translational order parameter for each particle.
*/

namespace freud { namespace order {

//! Parent class for Hexatic and Translational
/*!
 */
template<typename T> class HexaticTranslational
{
public:
    //! Constructor
    HexaticTranslational(T k) : m_k(k) {}

    //! Destructor
    virtual ~HexaticTranslational() {}

    T getK() const
    {
        return m_k;
    }

    //! Get a reference to the order parameter array
    const util::ManagedArray<std::complex<float>>& getOrder() const
    {
        return m_psi_array;
    }

protected:
    //! Compute the order parameter
    template<typename Func>
    void computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                        const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

    const T m_k;
    util::ManagedArray<std::complex<float>> m_psi_array; //!< psi array computed
};

//! Compute the translational order parameter for a set of points
/*!
 */
class Translational : public HexaticTranslational<float>
{
public:
    //! Constructor
    Translational(float k = 6);

    //! Destructor
    ~Translational();

    //! Compute the translational order parameter
    void compute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);
};

//! Compute the hexatic order parameter for a set of points
/*!
 */
class Hexatic : public HexaticTranslational<unsigned int>
{
public:
    //! Constructor
    Hexatic(unsigned int k = 6);

    //! Destructor
    ~Hexatic();

    //! Compute the hexatic order parameter
    void compute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);
};

}; }; // end namespace freud::order

#endif // HEXATIC_TRANSLATIONAL_H
