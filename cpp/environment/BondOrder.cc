// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondOrder.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"

/*! \file BondOrder.h
    \brief Compute the bond order diagram for the system of particles.
*/

namespace freud { namespace environment {

// namespace-level constant 2*pi for convenient use everywhere.
constexpr float TWO_PI = 2.0 * M_PI;

BondOrder::BondOrder(unsigned int n_bins_theta, unsigned int n_bins_phi)
    : BondHistogramCompute()
{
    // sanity checks, but this is actually kinda dumb if these values are 1
    if (n_bins_theta < 2)
        throw std::invalid_argument("BondOrder requires at least 2 bins in theta.");
    if (n_bins_phi < 2)
        throw std::invalid_argument("BondOrder requires at least 2 bins in phi.");
    // calculate dt, dp
    /*
    0 < \theta < 2PI; 0 < \phi < PI
    */
    float dt = 2.0 * M_PI / float(n_bins_theta);
    float dp = M_PI / float(n_bins_phi);
    // this shouldn't be able to happen, but it's always better to check
    if (dt > 2.0 * M_PI)
        throw std::invalid_argument("2PI must be greater than dt");
    if (dp > M_PI)
        throw std::invalid_argument("PI must be greater than dp");

    // precompute the surface area array
    m_sa_array.prepare({n_bins_theta, n_bins_phi});
    for (unsigned int i = 0; i < n_bins_theta; i++)
    {
        for (unsigned int j = 0; j < n_bins_phi; j++)
        {
            float phi = (float) j * dp;
            float sa = dt * (cos(phi) - cos(phi + dp));
            m_sa_array(i, j) = sa;
        }
    }
    BHAxes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_bins_theta, 0, TWO_PI));
    axes.push_back(std::make_shared<util::RegularAxis>(n_bins_phi, 0, M_PI));
    m_histogram = BondHistogram(axes);

    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);
}

void BondOrder::reduce()
{
    m_histogram.reset();
    m_bo_array.prepare(m_histogram.shape());

    m_histogram.reduceOverThreadsPerBin(m_local_histograms, [&] (size_t i) {
            m_bo_array[i] = m_histogram[i] / m_sa_array[i] / static_cast<float>(m_frame_counter);
        });
}

const util::ManagedArray<float> &BondOrder::getBondOrder()
{
    if (m_reduce == true)
    {
        reduce();
    }
    m_reduce = false;
    return m_bo_array;
}

void BondOrder::accumulate(
                    const locality::NeighborQuery* neighbor_query,
                    quat<float>* orientations, vec3<float>* query_points,
                    quat<float>* query_orientations, unsigned int n_query_points,
                    unsigned int mode,
                    const freud::locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs)
{
    // transform the mode from an integer to an enumerated type (enumerated in BondOrder.h)
    BondOrderMode b_mode = static_cast<BondOrderMode>(mode);

    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
    [=] (const freud::locality::NeighborBond& neighbor_bond)
    {
        vec3<float> ref_pos = neighbor_query->getPoints()[neighbor_bond.ref_id];
        quat<float>& ref_q = orientations[neighbor_bond.ref_id];
        vec3<float> v = m_box.wrap(query_points[neighbor_bond.id] - ref_pos);
        quat<float>& q = query_orientations[neighbor_bond.id];
        if (b_mode == obcd)
        {
            // give bond directions of neighboring particles rotated by the matrix
            // that takes the orientation of particle neighbor_bond.id to the orientation of
            // particle neighbor_bond.ref_id.
            v = rotate(conj(ref_q), v);
            v = rotate(q, v);
        }
        else if (b_mode == lbod)
        {
            // give bond directions of neighboring particles rotated into the
            // local orientation of the central particle.
            v = rotate(conj(ref_q), v);
        }
        else if (b_mode == oocd)
        {
            // give the directors of neighboring particles rotated into the local
            // orientation of the central particle. pick a (random vector)
            vec3<float> z(0, 0, 1);
            // rotate that vector by the orientation of the neighboring particle
            z = rotate(q, z);
            // get the direction of this vector with respect to the orientation of
            // the central particle
            v = rotate(conj(ref_q), z);
        }

        // NOTE that angles are defined in the "mathematical" way, rather than how
        // most physics textbooks do it. get theta (azimuthal angle), phi (polar
        // angle)
        float theta = atan2f(v.y, v.x); //-Pi..Pi

        theta = fmod(theta, 2 * M_PI);
        if (theta < 0)
        {
            theta += 2 * M_PI;
        }

        // NOTE that the below has replaced the commented out expression for phi.
        float phi = acos(v.z / sqrt(v.x * v.x + v.y * v.y + v.z * v.z)); // 0..Pi

        m_local_histograms(theta, phi);
    });
}

}; }; // end namespace freud::environment
