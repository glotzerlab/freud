// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#include <tbb/tbb.h>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondOrder.h"
#include "Index1D.h"
#include "NeighborComputeFunctional.h"
#include "NeighborBond.h"

using namespace std;
using namespace tbb;

/*! \file BondOrder.h
    \brief Compute the bond order diagram for the system of particles.
*/

namespace freud { namespace environment {

BondOrder::BondOrder(unsigned int n_bins_theta, unsigned int n_bins_phi)
    : m_box(box::Box()), m_n_bins_theta(n_bins_theta), m_n_bins_phi(n_bins_phi), m_frame_counter(0),
      m_reduce(true), m_local_bin_counts(n_bins_theta * n_bins_phi)
{
    // sanity checks, but this is actually kinda dumb if these values are 1
    if (m_n_bins_theta < 2)
        throw invalid_argument("BondOrder requires at least 2 bins in theta.");
    if (m_n_bins_phi < 2)
        throw invalid_argument("BondOrder requires at least 2 bins in phi.");
    // calculate dt, dp
    /*
    0 < \theta < 2PI; 0 < \phi < PI
    */
    m_dt = 2.0 * M_PI / float(m_n_bins_theta);
    m_dp = M_PI / float(m_n_bins_phi);
    // this shouldn't be able to happen, but it's always better to check
    if (m_dt > 2.0 * M_PI)
        throw invalid_argument("2PI must be greater than dt");
    if (m_dp > M_PI)
        throw invalid_argument("PI must be greater than dp");

    // precompute the bin center positions for t
    m_theta_array = std::shared_ptr<float>(new float[m_n_bins_theta], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_theta; i++)
    {
        float t = float(i) * m_dt;
        float nextt = float(i + 1) * m_dt;
        m_theta_array.get()[i] = ((t + nextt) / 2.0);
    }

    // precompute the bin center positions for p
    m_phi_array = std::shared_ptr<float>(new float[m_n_bins_phi], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_phi; i++)
    {
        float p = float(i) * m_dp;
        float nextp = float(i + 1) * m_dp;
        m_phi_array.get()[i] = ((p + nextp) / 2.0);
    }

    // precompute the surface area array
    m_sa_array = std::shared_ptr<float>(new float[m_n_bins_theta * m_n_bins_phi], std::default_delete<float[]>());
    memset((void*) m_sa_array.get(), 0, sizeof(float) * m_n_bins_theta * m_n_bins_phi);
    Index2D sa_i = Index2D(m_n_bins_theta, m_n_bins_phi);
    for (unsigned int i = 0; i < m_n_bins_theta; i++)
    {
        for (unsigned int j = 0; j < m_n_bins_phi; j++)
        {
            float phi = (float) j * m_dp;
            float sa = m_dt * (cos(phi) - cos(phi + m_dp));
            m_sa_array.get()[sa_i((int) i, (int) j)] = sa;
        }
    }

    // initialize the bin counts
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_n_bins_theta * m_n_bins_phi],
                                                 std::default_delete<unsigned int[]>());
    memset((void*) m_bin_counts.get(), 0, sizeof(unsigned int) * m_n_bins_theta * m_n_bins_phi);

    // initialize the bond order array
    m_bo_array = std::shared_ptr<float>(new float[m_n_bins_theta * m_n_bins_phi], std::default_delete<float[]>());
    memset((void*) m_bin_counts.get(), 0, sizeof(float) * m_n_bins_theta * m_n_bins_phi);
}

void BondOrder::reduceBondOrder()
{
    memset((void*) m_bo_array.get(), 0, sizeof(float) * m_n_bins_theta * m_n_bins_phi);
    memset((void*) m_bin_counts.get(), 0, sizeof(unsigned int) * m_n_bins_theta * m_n_bins_phi);
    parallel_for(blocked_range<size_t>(0, m_n_bins_theta), [=](const blocked_range<size_t>& r) {
        Index2D sa_i = Index2D(m_n_bins_theta, m_n_bins_phi);
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            for (size_t j = 0; j < m_n_bins_phi; j++)
            {
                for (util::ThreadStorage<unsigned int>::const_iterator local_bins
                     = m_local_bin_counts.begin();
                     local_bins != m_local_bin_counts.end(); ++local_bins)
                {
                    m_bin_counts.get()[sa_i((int) i, (int) j)] += (*local_bins)[sa_i((int) i, (int) j)];
                }
                m_bo_array.get()[sa_i((int) i, (int) j)]
                    = m_bin_counts.get()[sa_i((int) i, (int) j)] / m_sa_array.get()[sa_i((int) i, (int) j)];
            }
        }
    });
    Index2D sa_i = Index2D(m_n_bins_theta, m_n_bins_phi);
    for (unsigned int i = 0; i < m_n_bins_theta; i++)
    {
        for (unsigned int j = 0; j < m_n_bins_phi; j++)
        {
            m_bin_counts.get()[sa_i((int) i, (int) j)]
                = m_bin_counts.get()[sa_i((int) i, (int) j)] / (float) m_frame_counter;
            m_bo_array.get()[sa_i((int) i, (int) j)]
                = m_bo_array.get()[sa_i((int) i, (int) j)] / (float) m_frame_counter;
        }
    }
}

std::shared_ptr<float> BondOrder::getBondOrder()
{
    if (m_reduce == true)
    {
        reduceBondOrder();
    }
    m_reduce = false;
    return m_bo_array;
}

void BondOrder::reset()
{
    m_local_bin_counts.reset();
    // reset the frame counter
    m_frame_counter = 0;
    m_reduce = true;
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

    m_box = neighbor_query->getBox();
    // compute the order parameter

    float dt_inv = 1.0f / m_dt;
    float dp_inv = 1.0f / m_dp;
    Index2D sa_i = Index2D(m_n_bins_theta, m_n_bins_phi);

    freud::locality::loopOverNeighbors(neighbor_query, query_points, n_query_points, qargs, nlist,
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

        // bin the point
        float bin_theta = floorf(theta * dt_inv);
        float bin_phi = floorf(phi * dp_inv);
// fast float to int conversion with truncation
#ifdef __SSE2__
        unsigned int ibin_theta = _mm_cvtt_ss2si(_mm_load_ss(&bin_theta));
        unsigned int ibin_phi = _mm_cvtt_ss2si(_mm_load_ss(&bin_phi));
#else
            unsigned int ibin_theta = (unsigned int)(bin_theta);
            unsigned int ibin_phi = (unsigned int)(bin_phi);
#endif

        // increment the bin
        if ((ibin_theta < m_n_bins_theta) && (ibin_phi < m_n_bins_phi))
        {
            ++m_local_bin_counts.local()[sa_i(ibin_theta, ibin_phi)];
        }
    });

    // save the last computed number of particles
    m_frame_counter++;
    // flag to reduce
    m_reduce = true;
}

}; }; // end namespace freud::environment
