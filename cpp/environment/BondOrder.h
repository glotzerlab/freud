// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOND_ORDER_H
#define BOND_ORDER_H

#include "Box.h"
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
class BondOrder
{
public:
    //! Constructor
    BondOrder(unsigned int n_bins_theta, unsigned int n_bins_phi);

    //! Destructor
    ~BondOrder() {}

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Reset the bond order array to all zeros
    void reset();

    //! Accumulate the bond order
    void accumulate(const locality::NeighborQuery* neighbor_query,
                    quat<float>* orientations, vec3<float>* query_points,
                    quat<float>* query_orientations, unsigned int n_query_points,
                    unsigned int mode, const freud::locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    void reduceBondOrder();

    //! Get a reference to the last computed bond order
    const util::ManagedArray<float> &getBondOrder();

    //! Get a reference to the theta array
    const util::ManagedArray<float> &getTheta()
    {
        return m_theta_array;
    }

    //! Get a reference to the phi array
    const util::ManagedArray<float> &getPhi()
    {
        return m_phi_array;
    }

    unsigned int getNBinsTheta()
    {
        return m_n_bins_theta;
    }

    unsigned int getNBinsPhi()
    {
        return m_n_bins_phi;
    }

private:
    box::Box m_box; //!< Simulation box where the particles belong
    float m_dt;
    float m_dp;
    unsigned int m_n_bins_theta;  //!< number of bins for theta
    unsigned int m_n_bins_phi;    //!< number of bins for phi
    unsigned int m_frame_counter; //!< number of frames calculated
    bool m_reduce;                //!< Whether arrays need to be reduced across threads

    util::ManagedArray<unsigned int> m_bin_counts; //!< bin counts computed
    util::ManagedArray<float> m_bo_array;          //!< bond order array computed
    util::ManagedArray<float> m_sa_array;          //!< surface area array computed
    util::ManagedArray<float> m_theta_array;       //!< theta array computed
    util::ManagedArray<float> m_phi_array;         //!< phi order array computed
    util::ThreadStorage<unsigned int> m_local_bin_counts;
};

}; }; // end namespace freud::environment

#endif // BOND_ORDER_H
