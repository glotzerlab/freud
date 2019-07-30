// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>

#include "AngularSeparation.h"

using namespace std;
using namespace tbb;

/*! \file AngularSeparation.cc
    \brief Compute the angular separation for each particle.
*/

namespace freud { namespace environment {

AngularSeparation::AngularSeparation() : m_n_query_points(0), m_n_points(0), 
    m_n_global(0), m_n_equiv_orientations(0), m_tot_num_neigh(0) {}

AngularSeparation::~AngularSeparation() {}

float computeSeparationAngle(const quat<float> ref_q, const quat<float> q)
{
    quat<float> R = q * conj(ref_q);

    // if R.s is 1.0, but slightly greater due to round-off errors, handle that.
    if ((R.s - 1.0) < 1e-7 && (R.s - 1.0) > 0)
    {
        R.s = 1.0;
    }
    // if R.s is -1.0, but slightly less due to round-off errors, handle that.
    if ((R.s + 1.0) > -1e-7 && (R.s + 1.0) < 0)
    {
        R.s = -1.0;
    }

    float theta = 2.0 * acos(R.s);

    return theta;
}

// The set of all equivalent quaternions equiv_qs is the set that takes the particle as it
// is defined to some global reference orientation. Thus, to be safe, we must include
// a rotation by qconst as defined below when doing the calculation.
// Important: equiv_qs must include both q and -q, for all included quaternions
float computeMinSeparationAngle(const quat<float> ref_q, const quat<float> q, const quat<float>* equiv_qs,
                                unsigned int  n_equiv_quats)
{
    quat<float> qconst = equiv_qs[0];
    // here we undo a rotation represented by one of the equivalent orientations
    quat<float> qtemp = q * conj(qconst);

    // start with the quaternion before it has been rotated by equivalent rotations
    quat<float> min_quat = q;
    float min_angle = computeSeparationAngle(ref_q, q);

    // loop through all equivalent rotations and see if they have smaller angles with ref_q
    for (unsigned int i = 0; i <  n_equiv_quats; i++)
    {
        quat<float> qe = equiv_qs[i];
        quat<float> qtest = qtemp * qe;

        float angle_test = computeSeparationAngle(ref_q, qtest);

        if (angle_test < min_angle)
        {
            min_angle = angle_test;
            min_quat = qtest;
        }
    }

    return min_angle;
}

void AngularSeparation::computeNeighbor(const quat<float>* orientations,  unsigned int n_points,
                         const quat<float>* query_orientations, unsigned int n_query_points, 
                         const quat<float>* equiv_orientations, unsigned int n_equiv_orientations,
                         const freud::locality::NeighborList* nlist)
{
    nlist->validate(n_query_points, n_points);
    
    const size_t* neighbor_list(nlist->getNeighbors());
    // Get the maximum total number of bonds in the neighbor list
    const size_t tot_num_neigh = nlist->getNumBonds();

    // reallocate the output array if it is not the right size
    if (tot_num_neigh != m_tot_num_neigh)
    {
        m_neigh_ang_array = std::shared_ptr<float>(new float[tot_num_neigh], std::default_delete<float[]>());
    }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0, n_points), [=](const blocked_range<size_t>& r) {
        assert(orientations);
        assert(query_orientations);
        assert(equiv_orientations);
        assert(n_points > 0);
        assert(n_query_points > 0);
        assert(n_equiv_orientations > 0);

        size_t bond(nlist->find_first_index(r.begin()));
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            // m_neigh_ang_array.get()[i] = 0;
            quat<float> q = orientations[i];

            for (; bond < tot_num_neigh && neighbor_list[2 * bond] == i; ++bond)
            {
                const size_t j(neighbor_list[2 * bond + 1]);
                quat<float> query_q = query_orientations[j];

                float theta = computeMinSeparationAngle(q, query_q, equiv_orientations, n_equiv_orientations);
                m_neigh_ang_array.get()[bond] = theta;
            }
        }
    });
    // save the last computed number of particles
    m_n_query_points = n_query_points;
    // save the last computed number of reference particles
    m_n_points = n_points;
    // save the last computed number of equivalent quaternions
    m_n_equiv_orientations = n_equiv_orientations;
    // save the last computed number of total bonds
    m_tot_num_neigh = tot_num_neigh;
}

void AngularSeparation::computeGlobal(const quat<float>* global_orientations, unsigned int n_global, 
                       const quat<float>* orientations, unsigned int n_points, 
                       const quat<float>* equiv_orientations, unsigned int n_equiv_orientations)
{
    // reallocate the output array if it is not the right size
    if (n_points != m_n_points || n_global != m_n_global)
    {
        m_global_ang_array = std::shared_ptr<float>(new float[n_global * n_points], std::default_delete<float[]>());
    }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0, n_points), [=](const blocked_range<size_t>& r) {
        assert(global_orientations);
        assert(orientations);
        assert(equiv_orientations);
        assert(n_global > 0);
        assert(n_points > 0);
        assert( n_equiv_orientations > 0);

        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            quat<float> q = orientations[i];
            for (unsigned int j = 0; j < n_global; j++)
            {
                quat<float> global_q = global_orientations[j];
                float theta = computeMinSeparationAngle(q, global_q, equiv_orientations,  n_equiv_orientations);
                m_global_ang_array.get()[i * n_global + j] = theta;
            }
        }
    });
    // save the last computed number of orientations
    m_n_points = n_points;
    // save the last computed number of global orientations
    m_n_global = n_global;
    // save the last computed number of equivalent quaternions
    m_n_equiv_orientations =  n_equiv_orientations;
}

}; }; // end namespace freud::environment
