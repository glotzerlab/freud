// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "PMFTXYT.h"
#include <stdexcept>

/*! \file PMFTXYT.cc
    \brief Routines for computing potential of mean force and torque in XYT coordinates
*/

namespace freud { namespace pmft {

// namespace-level constant 2*pi for convenient use everywhere.
constexpr float TWO_PI = 2.0 * M_PI;

PMFTXYT::PMFTXYT(float x_max, float y_max, unsigned int n_x, unsigned int n_y, unsigned int n_t) : PMFT()
{
    if (n_x < 1)
        throw std::invalid_argument("PMFTXYT requires at least 1 bin in X.");
    if (n_y < 1)
        throw std::invalid_argument("PMFTXYT requires at least 1 bin in Y.");
    if (n_t < 1)
        throw std::invalid_argument("PMFTXYT requires at least 1 bin in T.");
    if (x_max < 0.0f)
        throw std::invalid_argument("PMFTXYT requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw std::invalid_argument("PMFTXYT requires that y_max must be positive.");

    // Compute Jacobian
    const float dx = 2.0 * x_max / float(n_x);
    const float dy = 2.0 * y_max / float(n_y);
    const float dt = TWO_PI / float(n_t);
    m_jacobian = dx * dy * dt;

    // create and populate the pcf_array
    m_pcf_array.prepare({n_x, n_y, n_t});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    BondHistogram::Axes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_x, -x_max, x_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_y, -y_max, y_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_t, 0, TWO_PI));
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYT::reduce()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    PMFT::reduce([jacobian_factor](size_t i) { return jacobian_factor; });
}

void PMFTXYT::accumulate(const locality::NeighborQuery* neighbor_query, float* orientations,
                         vec3<float>* query_points, float* query_orientations, unsigned int n_query_points,
                         const locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    neighbor_query->getBox().enforce2D();
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
                      [=](const freud::locality::NeighborBond& neighbor_bond) {
                          vec3<float> delta(bondVector(neighbor_bond, neighbor_query, query_points));

                          // rotate interparticle vector
                          vec2<float> myVec(delta.x, delta.y);
                          rotmat2<float> myMat
                              = rotmat2<float>::fromAngle(-orientations[neighbor_bond.point_idx]);
                          vec2<float> rotVec = myMat * myVec;
                          // calculate angle
                          float d_theta = atan2(-delta.y, -delta.x);
                          float t = query_orientations[neighbor_bond.query_point_idx] - d_theta;
                          // make sure that t is bounded between 0 and 2PI
                          t = fmod(t, TWO_PI);
                          if (t < 0)
                          {
                              t += TWO_PI;
                          }

                          m_local_histograms(rotVec.x, rotVec.y, t);
                      });
}
}; }; // end namespace freud::pmft
