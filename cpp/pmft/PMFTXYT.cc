// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Index1D.h"
#include "PMFTXYT.h"

using namespace std;
using namespace tbb;

/*! \file PMFTXYT.cc
    \brief Routines for computing potential of mean force and torque in XYT coordinates
*/

namespace freud { namespace pmft {

PMFTXYT::PMFTXYT(float x_max, float y_max, unsigned int n_x, unsigned int n_y, unsigned int n_t)
    : PMFT(), m_x_max(x_max), m_y_max(y_max), m_t_max(2.0 * M_PI), m_n_x(n_x), m_n_y(n_y), m_n_t(n_t)
{
    if (n_x < 1)
        throw invalid_argument("PMFTXYT requires at least 1 bin in X.");
    if (n_y < 1)
        throw invalid_argument("PMFTXYT requires at least 1 bin in Y.");
    if (n_t < 1)
        throw invalid_argument("PMFTXYT requires at least 1 bin in T.");
    if (x_max < 0.0f)
        throw invalid_argument("PMFTXYT requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw invalid_argument("PMFTXYT requires that y_max must be positive.");
    // calculate dx, dy, dt
    m_dx = 2.0 * m_x_max / float(m_n_x);
    m_dy = 2.0 * m_y_max / float(m_n_y);
    m_dt = m_t_max / float(m_n_t);

    if (m_dx > x_max)
        throw invalid_argument("PMFTXYT requires that dx is less than or equal to x_max.");
    if (m_dy > y_max)
        throw invalid_argument("PMFTXYT requires that dy is less than or equal to y_max.");
    if (m_dt > m_t_max)
        throw invalid_argument("PMFTXYT requires that dt is less than or equal to t_max.");

    m_jacobian = m_dx * m_dy * m_dt;

    // precompute the bin center positions for x
    m_x_array = precomputeAxisBinCenter(m_n_x, m_dx, m_x_max);
    // precompute the bin center positions for y
    m_y_array = precomputeAxisBinCenter(m_n_y, m_dy, m_y_max);
    // precompute the bin center positions for t
    m_t_array = precomputeAxisBinCenter(m_n_t, m_dt, 0);

    // create and populate the pcf_array
    m_pcf_array.prepare({m_n_x, m_n_y, m_n_t});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    util::Histogram::Axes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_x, -m_x_max, m_x_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_y, -m_y_max, m_y_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_t, 0, m_t_max));
    m_histogram = util::Histogram(axes);
    m_local_histograms = util::Histogram::ThreadLocalHistogram(m_histogram);

    // Set r_max
    m_r_max = sqrtf(m_x_max * m_x_max + m_y_max * m_y_max);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYT::reducePCF()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    reduce([jacobian_factor](size_t i) { return jacobian_factor; });
}

void PMFTXYT::accumulate(const locality::NeighborQuery* neighbor_query,
                         float* orientations, vec3<float>* query_points,
                         float* query_orientations, unsigned int n_query_points,
                         const locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
        [=](const freud::locality::NeighborBond& neighbor_bond) {
        vec3<float> ref = neighbor_query->getPoints()[neighbor_bond.ref_id];
        vec3<float> delta = m_box.wrap(query_points[neighbor_bond.id] - ref);
        //std::cout << "Ref: " << ref.x << ", " << ref.y << ", " << ref.z << std::endl;
        //std::cout << "Query: " << query_points[neighbor_bond.id].x << ", " << query_points[neighbor_bond.id].y << ", " << query_points[neighbor_bond.id].z << std::endl;
        //std::cout << "Delta: " << delta.x << ", " << delta.y << ", " << delta.z << std::endl;

        // rotate interparticle vector
        vec2<float> myVec(delta.x, delta.y);
        rotmat2<float> myMat = rotmat2<float>::fromAngle(-orientations[neighbor_bond.ref_id]);
        vec2<float> rotVec = myMat * myVec;
        // calculate angle
        float d_theta = atan2(-delta.y, -delta.x);
        float t = query_orientations[neighbor_bond.id] - d_theta;
        //std::cout << "The angle: " << t << std::endl;
        // make sure that t is bounded between 0 and 2PI
        t = fmod(t, 2 * M_PI);
        if (t < 0)
        {
            t += 2 * M_PI;
        }
        m_local_histograms(rotVec.x, rotVec.y, t);
    });
}
}; }; // end namespace freud::pmft
