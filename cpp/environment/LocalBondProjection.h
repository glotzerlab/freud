// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_BOND_PROJECTION_H
#define LOCAL_BOND_PROJECTION_H

#include "Box.h"
#include "NeighborQuery.h"
#include "NeighborList.h"
#include "VectorMath.h"
#include "ManagedArray.h"

/*! \file LocalBondProjection.h
    \brief Compute the projection of nearest neighbor bonds for each particle onto some
    set of reference vectors, defined in the particles' local reference frame.
*/

namespace freud { namespace environment {

//! Project the local bond onto all symmetrically equivalent vectors to proj_vec.
//! Return the maximal projection value.
float computeMaxProjection(const vec3<float> proj_vec, const vec3<float> local_bond,
                           const quat<float>* equiv_qs, unsigned int Nequiv);

class LocalBondProjection
{
public:
    //! Constructor
    LocalBondProjection();

    //! Destructor
    ~LocalBondProjection();

    //! Compute the maximal local bond projection
    void compute(const locality::NeighborQuery *nq,
        const quat<float>* orientations,
        const vec3<float>* query_points, unsigned int n_query_points,
        const vec3<float>* proj_vecs,  unsigned int n_proj,
        const quat<float>* equiv_orientations, unsigned int n_equiv_orientations,
        const freud::locality::NeighborList* nlist, locality::QueryArgs qargs);

    //! Get a reference to the last computed maximal local bond projection array
    const util::ManagedArray<float> &getProjections()
    {
        return m_local_bond_proj;
    }

    //! Get a reference to the last computed normalized maximal local bond projection array
    const util::ManagedArray<float> &getNormedProjections()
    {
        return m_local_bond_proj_norm;
    }

    unsigned int getNQueryPoints()
    {
        return m_n_query_points;
    }

    unsigned int getNPoints()
    {
        return m_n_points;
    }

    unsigned int getNproj()
    {
        return m_n_proj;
    }

    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Return a pointer to the NeighborList used in the last call to compute.
    locality::NeighborList *getNList()
    {
        return &m_nlist;
    }

private:
    box::Box m_box;               //!< Last used simulation box
    unsigned int m_n_query_points;            //!< Last number of particles computed
    unsigned int m_n_points;          //!< Last number of reference particles used for computation
    unsigned int m_n_proj;         //!< Last number of projection vectors used for computation
    unsigned int m_n_equiv_orientations;        //!< Last number of equivalent reference orientations used for computation
    unsigned int m_tot_num_neigh; //!< Last number of total bonds used for computation
    locality::NeighborList m_nlist; //!< The NeighborList used in the last call to compute.

    util::ManagedArray<float> m_local_bond_proj;      //!< Local bond projection array computed
    util::ManagedArray<float> m_local_bond_proj_norm; //!< Normalized local bond projection array computed
};

}; }; // end namespace freud::environment

#endif // LOCAL_BOND_PROJECTION_H
