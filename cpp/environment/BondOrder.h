// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOND_ORDER_H
#define BOND_ORDER_H

#include "Box.h"
#include "Histogram.h"
#include "BondHistogramCompute.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "ThreadStorage.h"
#include "VectorMath.h"
#include "ManagedArray.h"

/*! \file BondOrder.h
    \brief Compute the bond order diagram for the system of particles.
*/

namespace freud { namespace environment {

// this is needed for conversion of the type of bond order calculation to be made in accumulate.
typedef enum
{
    bod = 0,
    lbod = 1,
    obcd = 2,
    oocd = 3
} BondOrderMode;

//! Compute the bond order parameter for a set of points
/*!
 */
class BondOrder : public locality::BondHistogramCompute
{
public:
    //! Constructor
    BondOrder(unsigned int n_bins_theta, unsigned int n_bins_phi, BondOrderMode mode);

    //! Destructor
    ~BondOrder() {}

    //! Accumulate the bond order
    void accumulate(const locality::NeighborQuery* neighbor_query,
                    quat<float>* orientations, vec3<float>* query_points,
                    quat<float>* query_orientations, unsigned int n_query_points,
                    const freud::locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    virtual void reduce();

    //! Get a reference to the last computed bond order
    const util::ManagedArray<float> &getBondOrder();

private:
    util::ManagedArray<float> m_bo_array;          //!< bond order array computed
    util::ManagedArray<float> m_sa_array;          //!< surface area array computed
    BondOrderMode m_mode;                          //!< The mode to calculate with.
};

}; }; // end namespace freud::environment

#endif // BOND_ORDER_H
