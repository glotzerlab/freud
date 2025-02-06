// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOND_ORDER_H
#define BOND_ORDER_H

#include <memory>
#include <vector>

#include "BondHistogramCompute.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file BondOrder.h
    \brief Compute the bond order diagram for the system of particles.
*/

namespace freud { namespace environment {

// this is needed for conversion of the type of bond order calculation to be made in accumulate.
typedef enum // NOLINT(modernize-use-using)
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
    ~BondOrder() override = default;

    //! Accumulate the bond order
    void accumulate(const std::shared_ptr<locality::NeighborQuery>& neighbor_query, quat<float>* orientations,
                    vec3<float>* query_points, quat<float>* query_orientations, unsigned int n_query_points,
                    const std::shared_ptr<freud::locality::NeighborList>& nlist,
                    freud::locality::QueryArgs qargs);

    void reduce() override;
    void reset() override;
    std::vector<std::vector<float>> getBinCenters();

    //! Get a shared_ptr to the last computed bond order
    std::shared_ptr<const util::ManagedArray<float>> getBondOrder();

    BondOrderMode getMode() const
    {
        return m_mode;
    }

private:
    std::shared_ptr<util::ManagedArray<float>> m_bo_array; //!< bond order array computed
    std::shared_ptr<util::ManagedArray<float>> m_sa_array; //!< surface area array computed
    BondOrderMode m_mode;                                  //!< The mode to calculate with.
};

}; }; // end namespace freud::environment

#endif // BOND_ORDER_H
