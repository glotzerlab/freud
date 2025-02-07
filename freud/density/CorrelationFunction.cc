// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondHistogramCompute.h"
#include "CorrelationFunction.h"
#include "Histogram.h"
#include "NeighborBond.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file CorrelationFunction.cc
    \brief Generic pairwise correlation functions.
*/

namespace freud { namespace density {

CorrelationFunction::CorrelationFunction(unsigned int bins, float r_max) : BondHistogramCompute()
{
    if (bins == 0)
    {
        throw std::invalid_argument("CorrelationFunction  requires a nonzero number of bins.");
    }
    if (r_max <= 0)
    {
        throw std::invalid_argument("CorrelationFunction requires r_max to be positive.");
    }

    // We must construct two separate histograms, one for the counts and one
    // for the actual correlation function. The counts are used to normalize
    // the correlation function. The histograms can share the same set of axes.
    const auto axes = util::Axes {std::make_shared<util::RegularAxis>(bins, 0, r_max)};
    m_histogram = util::Histogram<unsigned int>(axes);
    m_local_histograms = util::Histogram<unsigned int>::ThreadLocalHistogram(m_histogram);

    m_correlation_function = util::Histogram<std::complex<double>>(axes);
    m_local_correlation_function = CFThreadHistogram(m_correlation_function);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void CorrelationFunction::reduce()
{
    // Reduce the bin counts over all threads, then use them to normalize
    m_histogram.reduceOverThreads(m_local_histograms);
    m_correlation_function.reduceOverThreadsPerBin(m_local_correlation_function, [&](size_t i) {
        if (m_histogram[i] != 0)
        {
            m_correlation_function[i] /= m_histogram[i];
        }
    });
}

void CorrelationFunction::reset()
{
    BondHistogramCompute::reset();
    m_correlation_function = util::Histogram<std::complex<double>>(m_histogram.getAxes());
    m_local_correlation_function.reset();
}

// Define an overloaded pair of product functions to deal with complex conjugation if necessary.
inline std::complex<double> product(std::complex<double> x, std::complex<double> y)
{
    return std::conj(x) * y;
}

inline double product(double x, double y)
{
    return x * y;
}

void CorrelationFunction::accumulate(const std::shared_ptr<freud::locality::NeighborQuery>& neighbor_query,
                                     const std::complex<double>* values, const vec3<float>* query_points,
                                     const std::complex<double>* query_values, unsigned int n_query_points,
                                     const std::shared_ptr<freud::locality::NeighborList>& nlist,
                                     const freud::locality::QueryArgs& qargs)
{
    accumulateGeneral(
        neighbor_query, query_points, n_query_points, nlist, qargs,
        [&](const freud::locality::NeighborBond& neighbor_bond) {
            const size_t value_bin = m_histogram.bin({neighbor_bond.getDistance()});
            m_local_histograms.increment(value_bin);
            m_local_correlation_function.increment(
                value_bin,
                product(values[neighbor_bond.getPointIdx()], query_values[neighbor_bond.getQueryPointIdx()]));
        });
}

class CorrelationFunction;

}; }; // end namespace freud::density
