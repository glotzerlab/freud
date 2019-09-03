// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#include <tbb/tbb.h>
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <iostream>

#include "NeighborBond.h"
#include "RDF.h"

using namespace std;
using namespace tbb;

/*! \file RDF.cc
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {

RDF::RDF(float r_max, float dr, float r_min) : m_r_max(r_max), m_r_min(r_min), m_dr(dr), m_box(box::Box()), m_frame_counter(0), m_n_points(0),
    m_n_query_points(0), m_reduce(true)
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
    m_pcf_array = util::makeEmptyArray<float>(m_nbins);
    m_avg_counts = util::makeEmptyArray<float>(m_nbins);
    m_N_r_array = util::makeEmptyArray<float>(m_nbins);


    std::vector<std::shared_ptr<util::Axis> > axes;
    axes.push_back(std::make_shared<util::RegularAxis>(m_nbins, m_r_min, m_r_max));
    m_bin_counts = util::Histogram(axes);
    m_local_bin_counts = util::Histogram::ThreadLocalHistogram(m_bin_counts);

    // precompute the bin center positions and cell volumes
    m_r_array = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    m_vol_array = util::makeEmptyArray<float>(m_nbins);
    m_vol_array2D = util::makeEmptyArray<float>(m_nbins);
    m_vol_array3D = util::makeEmptyArray<float>(m_nbins);

    for (unsigned int i = 0; i < m_nbins; i++)
    {
        float r = float(i) * m_dr + m_r_min;
        float nextr = float(i + 1) * m_dr + m_r_min;
        m_r_array.get()[i] = 2.0f / 3.0f * (nextr * nextr * nextr - r * r * r) / (nextr * nextr - r * r);
        m_vol_array2D.get()[i] = M_PI * (nextr * nextr - r * r);
        m_vol_array3D.get()[i] = 4.0f / 3.0f * M_PI * (nextr * nextr * nextr - r * r * r);
    }
} // end RDF::RDF

//! \internal
//! CumulativeCount class to perform a parallel reduce to get the cumulative count for each histogram bin
class CumulativeCount
{
private:
    float m_sum;
    float* m_N_r_array;
    float* m_avg_counts;

public:
    CumulativeCount(float* N_r_array, float* avg_counts)
        : m_sum(0), m_N_r_array(N_r_array), m_avg_counts(avg_counts)
    {}
    float get_sum() const
    {
        return m_sum;
    }
    template<typename Tag> void operator()(const blocked_range<size_t>& r, Tag)
    {
        float temp = m_sum;
        for (size_t i = r.begin(); i < r.end(); i++)
        {
            temp = temp + m_avg_counts[i];
            if (Tag::is_final_scan())
                m_N_r_array[i] = temp;
        }
        m_sum = temp;
    }
    CumulativeCount(CumulativeCount& b, split)
        : m_sum(0), m_N_r_array(b.m_N_r_array), m_avg_counts(b.m_avg_counts)
    {}
    void reverse_join(CumulativeCount& a)
    {
        m_sum = a.m_sum + m_sum;
    }
    void assign(CumulativeCount& b)
    {
        m_sum = b.m_sum;
    }
};

//! \internal
//! helper function to reduce the thread specific arrays into one array
void RDF::reduceRDF()
{
    m_bin_counts.reset();
    memset((void*) m_avg_counts.get(), 0, sizeof(float) * m_nbins);
    // now compute the rdf
    float ndens = float(m_n_query_points) / m_box.getVolume();
    m_pcf_array.get()[0] = 0.0f;
    m_N_r_array.get()[0] = 0.0f;
    m_N_r_array.get()[1] = 0.0f;
    if (m_box.is2D())
        m_vol_array = m_vol_array2D;
    else
        m_vol_array = m_vol_array3D;
    // Now compute the rdf. We skip the 0 bin since there can't be anything there.
    m_bin_counts.reduceOverThreadsPerParticle(m_local_bin_counts, [this, &ndens](unsigned int i) {
        this->m_avg_counts.get()[i] = static_cast<float>(this->m_bin_counts.getBinCounts()[i]) / this->m_n_points;
        this->m_pcf_array.get()[i] = this->m_avg_counts.get()[i] / m_vol_array.get()[i] / ndens;
    });

    CumulativeCount myN_r(m_N_r_array.get(), m_avg_counts.get());
    parallel_scan(blocked_range<size_t>(0, m_nbins), myN_r);

    for (unsigned int i = 0; i < m_nbins; i++)
    {
        m_pcf_array.get()[i] /= m_frame_counter;
        m_N_r_array.get()[i] /= m_frame_counter;
    }
}

//! get a reference to the histogram bin centers array
std::shared_ptr<float> RDF::getR()
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
    m_local_bin_counts.reset();
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

    m_box = neighbor_query->getBox();
    locality::loopOverNeighbors(neighbor_query, query_points, n_query_points, qargs, nlist,
           [=](const freud::locality::NeighborBond& neighbor_bond) {
        if (neighbor_bond.distance < m_r_max && neighbor_bond.distance > m_r_min)
        {
            m_local_bin_counts.local()(neighbor_bond.distance);
        }
    });
    // We ignore anything binned in the zero bin to avoid any confusion.
    // USING THIS CODE MAKES THE CUMULATIVE COUNT RIGHT, BUT IT MAKES THE RDF WRONG. NEED TO THINK ABOUT EXACTLY WHY TO MAKE SURE I UNDERSTAND THE RIGHT WAY TO DO IT.
    //for (auto it = m_local_bin_counts.begin(); it != m_local_bin_counts.end(); it++)
    //{
        //(*it).m_bin_counts[0] = 0;
    //}
    m_frame_counter++;
    m_n_points = neighbor_query->getNPoints();
    m_n_query_points = n_query_points;
    // flag to reduce
    m_reduce = true;
}

}; }; // end namespace freud::density
