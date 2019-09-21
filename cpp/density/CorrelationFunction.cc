// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#include <tbb/tbb.h>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "CorrelationFunction.h"
#include "NeighborComputeFunctional.h"
#include "NeighborBond.h"

/*! \file CorrelationFunction.cc
    \brief Generic pairwise correlation functions.
*/

namespace freud { namespace density {

template<typename T>
CorrelationFunction<T>::CorrelationFunction(float r_max, float dr)
    : m_box(box::Box()), m_r_max(r_max), m_dr(dr), m_frame_counter(0), m_reduce(true)
{
    if (dr <= 0.0f)
        throw std::invalid_argument("CorrelationFunction requires dr to be positive.");
    if (r_max <= 0.0f)
        throw std::invalid_argument("CorrelationFunction requires r_max to be positive.");
    if (dr > r_max)
        throw std::invalid_argument("CorrelationFunction requires dr must be less than or equal to r_max.");

    m_nbins = int(floorf(m_r_max / m_dr));
    assert(m_nbins > 0);
    m_rdf_array.prepare(m_nbins);
    // Less efficient: initialize each bin sequentially using default ctor
    for (size_t i(0); i < m_nbins; ++i)
        m_rdf_array[i] = T();
    m_bin_counts.prepare(m_nbins);

    // precompute the bin center positions
    m_r_array.prepare(m_nbins);
    for (unsigned int i = 0; i < m_nbins; i++)
    {
        float r = float(i) * m_dr;
        float nextr = float(i + 1) * m_dr;
        m_r_array[i] = 2.0f / 3.0f * (nextr * nextr * nextr - r * r * r) / (nextr * nextr - r * r);
    }
    m_local_bin_counts.resize(m_nbins);
    m_local_rdf_array.resize(m_nbins);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
template<typename T> void CorrelationFunction<T>::reduceCorrelationFunction()
{
    m_bin_counts.prepare(m_nbins);
    for (size_t i(0); i < m_nbins; ++i)
        m_rdf_array.get()[i] = T();
    // now compute the rdf
    util::forLoopWrapper(0, m_nbins, [=](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            for (util::ThreadStorage<unsigned int>::const_iterator local_bins = m_local_bin_counts.begin();
                 local_bins != m_local_bin_counts.end(); ++local_bins)
            {
                m_bin_counts[i] += (*local_bins)[i];
            }
            for (typename util::ThreadStorage<T>::const_iterator local_rdf = m_local_rdf_array.begin();
                 local_rdf != m_local_rdf_array.end(); ++local_rdf)
            {
                m_rdf_array[i] += (*local_rdf)[i];
            }
            if (m_bin_counts[i])
            {
                m_rdf_array[i] /= m_bin_counts[i];
            }
        }
    });
}

//! Get a reference to the RDF array
template<typename T>
const util::ManagedArray<T> &CorrelationFunction<T>::getRDF()
{
    if (m_reduce == true)
    {
        reduceCorrelationFunction();
    }
    m_reduce = false;
    return m_rdf_array;
}

//! \internal
/*! \brief Function to reset the PCF array if needed e.g. calculating between new particle types
 */
template<typename T> void CorrelationFunction<T>::reset()
{
    // zero the bin counts for totaling
    m_local_rdf_array.reset();
    m_local_bin_counts.reset();
    // reset the frame counter
    m_frame_counter = 0;
    m_reduce = true;
}

template<typename T>
void CorrelationFunction<T>::accumulate(const freud::locality::NeighborQuery* neighbor_query, const T* values,
                                        const vec3<float>* query_points, const T* query_values,
                                        unsigned int n_query_points, const freud::locality::NeighborList* nlist,
                                        freud::locality::QueryArgs qargs)
{
    m_box = neighbor_query->getBox();
    float dr_inv = 1.0f / m_dr;
    freud::locality::loopOverNeighbors(neighbor_query, query_points, n_query_points, qargs, nlist,
    [=](const freud::locality::NeighborBond& neighbor_bond)
        {
            // bin that r
            float binr = neighbor_bond.distance * dr_inv;
            // fast float to int conversion with truncation
            #ifdef __SSE2__
            unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
            #else
            unsigned int bin = (unsigned int)(binr);
            #endif

            if (bin < m_nbins)
            {
                ++m_local_bin_counts.local()[bin];
                m_local_rdf_array.local()[bin] += values[neighbor_bond.ref_id] * query_values[neighbor_bond.id];
            }
        }
    );
    m_frame_counter += 1;
    m_reduce = true;
}

template class CorrelationFunction<std::complex<double>>;
template class CorrelationFunction<double>;

}; }; // end namespace freud::density
