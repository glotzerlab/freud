// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "Box.h"
#include "RDF.h"
#include "StructureFactor.h"
#include "utils.h"

/*! \file StructureFactor.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

StructureFactor::StructureFactor(unsigned int bins, float k_max, float k_min)
{
    if (bins == 0)
        throw std::invalid_argument("StructureFactor requires a nonzero number of bins.");
    if (k_max <= 0.0f)
        throw std::invalid_argument("StructureFactor requires k_max to be positive.");
    if (k_max <= k_min)
        throw std::invalid_argument("StructureFactor requires that k_max must be greater than k_min.");

    // Construct the Histogram object that will be used to track the structure factor
    auto axes = StructureFactorHistogram::Axes {std::make_shared<util::RegularAxis>(bins, k_min, k_max)};
    m_histogram = StructureFactorHistogram(axes);
    m_local_histograms = StructureFactorHistogram::ThreadLocalHistogram(m_histogram);
    m_min_valid_k = std::numeric_limits<float>::infinity();
    m_structure_factor.prepare(bins);
}

void StructureFactor::accumulate(const freud::locality::NeighborQuery* neighbor_query,
                                 const vec3<float>* query_points, unsigned int n_query_points,
                                 const freud::locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    auto const& box = neighbor_query->getBox();

    // Normalization is 4 * pi * N / V
    auto const normalization = 2 * freud::constants::TWO_PI * n_query_points / box.getVolume();

    // The RDF r_max should be just less than half of the smallest side length of the box
    auto const box_L = box.getL();
    auto const min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    auto const r_max = std::nextafter(0.5f * min_box_length, 0.0f);

    // The minimum k value of validity for the RDF Fourier Transform method is 4 * pi / L, where L is the
    // smallest side length. This is equal to 2 * pi / r_max.
    m_min_valid_k = std::min(m_min_valid_k, freud::constants::TWO_PI / r_max);

    auto const rdf_bins = 1000;
    static_assert(rdf_bins % 2 == 0, "RDF bins must be even for the Simpson's rule calculation.");
    auto rdf = freud::density::RDF(rdf_bins, r_max);
    rdf.accumulate(neighbor_query, query_points, n_query_points, nlist, qargs);
    auto const& rdf_values = rdf.getRDF();

    util::forLoopWrapper(0, m_histogram.getAxisSizes()[0], [&](size_t begin_k, size_t end_k) {
        for (size_t k = begin_k; k < end_k; k++)
        {
            // Integrate using Simpson's rule
            auto integral = 0.0;

            // Simpson's rule uses prefactors 1, 4, 2, 4, 2, ..., 4, 1
            auto simpson_prefactor = [=](size_t bin) {
                if (bin == 0 || bin == rdf_bins - 1)
                {
                    return 1;
                }
                else if (bin % 2 == 0)
                {
                    return 2;
                }
                else
                {
                    return 4;
                }
            };

            auto const k_bin_edges = m_histogram.getBinEdges()[0];

            auto integrand = [&](size_t k, size_t rdf_index) {
                auto r_value = rdf.getBinEdges()[0][rdf_index];
                auto rdf_value = rdf.getRDF()[rdf_index];
                auto k_value = k_bin_edges[k];
                return r_value * r_value * (rdf_value - 1) * util::sinc(k_value * r_value);
            };

            for (size_t rdf_index = 0; rdf_index < rdf_bins; rdf_index++)
            {
                integral += simpson_prefactor(rdf_index) * integrand(k, rdf_index);
            }
            auto const dk = (k_bin_edges.back() - k_bin_edges.front()) / k_bin_edges.size();
            integral *= dk / 3;
            m_local_histograms.increment(k, integral);
        }
    });

    m_reduce = true;
}

const util::ManagedArray<float>& StructureFactor::getStructureFactor()
{
    if (m_reduce)
    {
        m_local_histograms.reduceInto(m_structure_factor);
    }
    return m_structure_factor;
}

}; }; // namespace freud::diffraction
