// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#include <tbb/tbb.h>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "NeighborBond.h"
#include "RDF.h"

using namespace std;
using namespace tbb;

/*! \file RDF.cc
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {

RDF::RDF(float r_max, float dr, float r_min) : m_box(box::Box()), m_frame_counter(0), m_n_points(0),
    m_n_query_points(0), m_reduce(true), m_r_max(r_max), m_r_min(r_min), m_dr(dr)
{
    if (dr <= 0.0f)
        throw invalid_argument("RDF requires dr to be positive.");
    if (r_max <= 0.0f)
        throw invalid_argument("RDF requires r_max to be positive.");
    if (dr > r_max)
        throw invalid_argument("RDF requires dr must be less than or equal to r_max.");
    if (r_max <= r_min)
        throw invalid_argument("RDF requires that r_max must be greater than r_min.");
    if (r_max - r_min < dr)
        throw invalid_argument("RDF requires that the range (r_max-r_min) must be greater than dr.");

    m_nbins = int(floorf((m_r_max - m_r_min) / m_dr));
    assert(m_nbins > 0);
    util::Histogram::Axes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(m_nbins, m_r_min, m_r_max));
    m_histogram = util::Histogram(axes);

    // precompute the bin center positions and cell volumes
    m_r_array.prepare(m_nbins);
    m_vol_array.prepare(m_nbins);
    m_vol_array2D.prepare(m_nbins);
    m_vol_array3D.prepare(m_nbins);

    for (unsigned int i = 0; i < m_nbins; i++)
    {
        float r = float(i) * m_dr + m_r_min;
        float nextr = float(i + 1) * m_dr + m_r_min;
        m_r_array.get()[i] = (r + nextr)/2;
        m_vol_array2D.get()[i] = M_PI * (nextr * nextr - r * r);
        m_vol_array3D.get()[i] = 4.0f / 3.0f * M_PI * (nextr * nextr * nextr - r * r * r);
    }
    m_local_histograms = util::Histogram::ThreadLocalHistogram(m_histogram);
} // end RDF::RDF

//! \internal
//! helper function to reduce the thread specific arrays into one array
void RDF::reduce()
{
    m_pcf_array.prepare(m_nbins);
    m_histogram.reset();
    m_avg_counts.prepare(m_nbins);
    m_N_r_array.prepare(m_nbins);

    // now compute the rdf
    float ndens = float(m_n_query_points) / m_box.getVolume();
    if (m_box.is2D())
        m_vol_array = m_vol_array2D;
    else
        m_vol_array = m_vol_array3D;
    // now compute the rdf
    m_histogram.reduceOverThreadsPerParticle(m_local_histograms,
            [this, &ndens] (size_t i) {
            m_avg_counts[i] = static_cast<float>(m_histogram[i]) / m_n_points;
            m_pcf_array[i] = m_avg_counts[i] / m_vol_array[i] / ndens;
            });

    m_N_r_array.get()[0] = m_avg_counts.get()[0];
    for (unsigned int i = 1; i < m_nbins; i++)
    {
        m_N_r_array.get()[i] = m_N_r_array.get()[i-1] + m_avg_counts.get()[i];
    }

    for (unsigned int i = 0; i < m_nbins; i++)
    {
        m_pcf_array[i] /= m_frame_counter;
        m_N_r_array.get()[i] /= m_frame_counter;
    }
}

//! get a reference to the histogram bin centers array
const util::ManagedArray<float> &RDF::getR()
{
    return m_r_array;
}

//! Get number of bins
unsigned int RDF::getNBins()
{
    return m_nbins;
}

//! \internal
/*! \brief Function to reset the rdf array if needed e.g. calculating between new particle types
 */
void RDF::reset()
{
    m_local_histograms.reset();
    this->m_frame_counter = 0;
    this->m_reduce = true;
}

//! \internal
/*! \brief Function to accumulate the given points to the histogram in memory
 */
void RDF::accumulate(const freud::locality::NeighborQuery* neighbor_query,
                    const vec3<float>* query_points, unsigned int n_query_points,
                    const freud::locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    m_n_query_points = n_query_points;
    m_n_points = neighbor_query->getNPoints();

    assert(neighbor_query);
    assert(query_points);
    assert(m_n_points > 0);
    assert(n_query_points > 0);

    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
        [=](const freud::locality::NeighborBond& neighbor_bond) {
        if (neighbor_bond.distance < m_r_max && neighbor_bond.distance > m_r_min)
        {
            m_local_histograms(neighbor_bond.distance);
        }
    });
}
}; }; // end namespace freud::density
