// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "PMFTXY2D.h"

/*! \file PMFTXY2D.cc
    \brief Routines for computing 2D potential of mean force in XY coordinates
*/

namespace freud { namespace pmft {

PMFTXY2D::PMFTXY2D(float x_max, float y_max, unsigned int n_x, unsigned int n_y)
    : PMFT()
{
    if (n_x < 1)
        throw std::invalid_argument("PMFTXY2D requires at least 1 bin in X.");
    if (n_y < 1)
        throw std::invalid_argument("PMFTXY2D requires at least 1 bin in Y.");
    if (x_max < 0.0f)
        throw std::invalid_argument("PMFTXY2D requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw std::invalid_argument("PMFTXY2D requires that y_max must be positive.");

    // Compute jacobian
    float dx = 2.0 * x_max / float(n_x);
    float dy = 2.0 * y_max / float(n_y);
    m_jacobian = dx * dy;

    // create the pcf_array
    m_pcf_array.prepare({n_x, n_y});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    BHAxes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_x, -x_max, x_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_y, -y_max, y_max));
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXY2D::reducePCF()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    reduce([jacobian_factor](size_t i) { return jacobian_factor; });
}

//! \internal
/*! \brief Helper functionto direct the calculation to the correct helper class
 */
void PMFTXY2D::accumulate(const locality::NeighborQuery* neighbor_query,
                          float* orientations, vec3<float>* query_points,
                          unsigned int n_query_points,
                          const locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
        [=](const freud::locality::NeighborBond& neighbor_bond) {
        vec3<float> ref = neighbor_query->getPoints()[neighbor_bond.point_idx];
        vec3<float> delta = this->m_box.wrap(query_points[neighbor_bond.query_point_idx] - ref);

        // rotate interparticle vector
        vec2<float> myVec(delta.x, delta.y);
        rotmat2<float> myMat = rotmat2<float>::fromAngle(-orientations[neighbor_bond.point_idx]);
        vec2<float> rotVec = myMat * myVec;

        m_local_histograms(rotVec.x, rotVec.y);
    });
}

}; }; // end namespace freud::pmft
