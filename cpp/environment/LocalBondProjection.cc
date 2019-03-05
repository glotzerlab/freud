// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "LocalBondProjection.h"

using namespace std;
using namespace tbb;

/*! \file LocalBondProjection.h
    \brief Compute the projection of nearest neighbor bonds for each particle onto some
    set of reference vectors, defined in the particles' local reference frame.
*/

namespace freud { namespace environment {

LocalBondProjection::LocalBondProjection()
    : m_Np(0), m_Nref(0), m_Nproj(0), m_Nequiv(0), m_tot_num_neigh(0)
    {
    }

LocalBondProjection::~LocalBondProjection()
    {
    }

// The set of all equivalent quaternions equiv_qs is the set that takes the particle as it
// is defined to some global reference orientation. Thus, to be safe, we must include
// a rotation by qconst as defined below when doing the calculation.
// IMPT: equiv_qs does NOT have to include both q and -q, for all included quaternions.
// Rather, equiv_qs SHOULD contain the identity, and have the same length as the order of
// the chiral symmetry group of the particle shape.
// q and -q effect the same rotation on vectors, and here we just use equiv_quats to
// find all symmetrically equivalent vectors to proj_vec.
float computeMaxProjection(const vec3<float> proj_vec, const vec3<float> local_bond,
    const quat<float> *equiv_qs, unsigned int Nequiv)
    {
    quat<float> qconst = equiv_qs[0];

    // start with the reference vector before it has been rotated by equivalent quaternions
    vec3<float> max_proj_vec = proj_vec;
    float max_proj = dot(proj_vec, local_bond);

    // loop through all equivalent rotations and see if they have a larger projection onto local_bond
    for (unsigned int i = 0; i<Nequiv; i++)
        {
        quat<float> qe = equiv_qs[i];
        // here we undo a rotation represented by one of the equivalent orientations
        quat<float> qtest = conj(qconst)*qe;
        vec3<float> equiv_proj_vec = rotate(qtest, proj_vec);

        float proj_test = dot(equiv_proj_vec, local_bond);

        if (proj_test > max_proj)
            {
            max_proj = proj_test;
            max_proj_vec = equiv_proj_vec;
            }

        }

    return max_proj;

    }

void LocalBondProjection::compute(
            box::Box& box,
            const freud::locality::NeighborList *nlist,
            const vec3<float> *pos,
            const vec3<float> *ref_pos,
            const quat<float> *ref_ors,
            const quat<float> *ref_equiv_ors,
            const vec3<float> *proj_vecs,
            unsigned int Np,
            unsigned int Nref,
            unsigned int Nequiv,
            unsigned int Nproj)

    {
    assert(pos);
    assert(ref_pos);
    assert(ref_ors);
    assert(ref_equiv_ors);
    assert(proj_vecs);
    assert(Np > 0);
    assert(Nref > 0);
    assert(Nequiv > 0);
    assert(Nproj > 0);

    nlist->validate(Nref, Np);
    const size_t *neighbor_list(nlist->getNeighbors());
    // Get the maximum total number of bonds in the neighbor list
    const size_t tot_num_neigh = nlist->getNumBonds();

    // reallocate the output array if it is not the right size
    if (tot_num_neigh != m_tot_num_neigh || Nproj != m_Nproj)
        {
        m_local_bond_proj = std::shared_ptr<float>(new float [tot_num_neigh*Nproj], std::default_delete<float[]>());
        m_local_bond_proj_norm = std::shared_ptr<float>(new float [tot_num_neigh*Nproj], std::default_delete<float[]>());
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Nref),
    [=] (const blocked_range<size_t>& r)
    {
    size_t bond(nlist->find_first_index(r.begin()));
    for(size_t i=r.begin(); i!=r.end(); ++i)
        {
        vec3<float> p = ref_pos[i];
        quat<float> q = ref_ors[i];

        for (; bond < tot_num_neigh && neighbor_list[2*bond] == i; ++bond)
            {
            const size_t j(neighbor_list[2*bond + 1]);
            // compute bond vector between the two particles
            vec3<float> delta = box.wrap(pos[j] - p);
            vec3<float> local_bond(delta);
            // rotate bond vector into the local frame of particle p
            local_bond = rotate(conj(q), local_bond);
            // store the length of this local bond
            float local_bond_len = sqrt(dot(local_bond, local_bond));

            for (unsigned int k=0; k<Nproj; k++)
                {
                vec3<float> proj_vec = proj_vecs[k];
                float max_proj = computeMaxProjection(proj_vec, local_bond, ref_equiv_ors, Nequiv);
                m_local_bond_proj.get()[bond*Nproj+k] = max_proj;
                m_local_bond_proj_norm.get()[bond*Nproj+k] = max_proj/local_bond_len;
                }
            }
        }

    });

    // save the last computed box
    m_box = box;
    // save the last computed number of particles
    m_Np = Np;
    // save the last computed number of reference particles
    m_Nref = Nref;
    // save the last computed number of equivalent quaternions
    m_Nequiv = Nequiv;
    // save the last computed number of reference projection vectors
    m_Nproj = Nproj;
    // save the last computed number of total bonds
    m_tot_num_neigh = tot_num_neigh;
    }

}; }; // end namespace freud::environment
