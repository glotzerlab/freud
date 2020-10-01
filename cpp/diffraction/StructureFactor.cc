// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "Box.h"
#include "ManagedArray.h"
#include "RDF.h"
#include "StructureFactor.h"
#include "utils.h"

/*! \file StructureFactor.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

StructureFactor::StructureFactor(unsigned int bins, float k_max, float k_min, bool direct) : m_direct(direct)
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
    if (m_direct)
    {
        accumulateDirect(neighbor_query, query_points, n_query_points, nlist, qargs);
    }
    else
    {
        accumulateRDF(neighbor_query, query_points, n_query_points, nlist, qargs);
    }
    m_reduce = true;
}

void StructureFactor::accumulateDirect(const freud::locality::NeighborQuery* neighbor_query,
                                       const vec3<float>* query_points, unsigned int n_query_points,
                                       const freud::locality::NeighborList* nlist,
                                       freud::locality::QueryArgs qargs)
{
    auto const& box = neighbor_query->getBox();
    auto distances = std::vector<float>(n_query_points * n_query_points);
    box.computeAllDistances(query_points, n_query_points, query_points, n_query_points, distances.data());

    auto const k_bin_centers = m_histogram.getBinCenters()[0];

    util::forLoopWrapper(0, m_histogram.getAxisSizes()[0], [&](size_t begin_k_index, size_t end_k_index) {
        auto sinc_values = std::vector<float>(n_query_points * n_query_points);
        for (size_t k_index = begin_k_index; k_index < end_k_index; ++k_index)
        {
            auto const k = k_bin_centers[k_index];
            double S_k = 0.0;
            std::for_each(distances.cbegin(), distances.cend(),
                          [&S_k, k](float const& distance) { S_k += util::sinc(k * distance); });
            S_k /= static_cast<double>(n_query_points);
            m_local_histograms.increment(k_index, S_k);
        };
    });
}

void StructureFactor::accumulateRDF(const freud::locality::NeighborQuery* neighbor_query,
                                    const vec3<float>* query_points, unsigned int n_query_points,
                                    const freud::locality::NeighborList* nlist,
                                    freud::locality::QueryArgs qargs)
{
    auto const& box = neighbor_query->getBox();

    // Normalization is 4 * pi * N / V
    m_normalization = 2.0 * freud::constants::TWO_PI * static_cast<float>(n_query_points) / box.getVolume();

    // The RDF r_max should be just less than half of the smallest side length of the box
    auto const box_L = box.getL();
    auto const min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    auto const r_max = std::nextafter(0.5f * min_box_length, 0.0f);

    // The minimum k value of validity for the RDF Fourier Transform method is 4 * pi / L, where L is the
    // smallest side length. This is equal to 2 * pi / r_max.
    m_min_valid_k = std::min(m_min_valid_k, freud::constants::TWO_PI / r_max);

    auto const rdf_bins = 1001;
    static_assert(rdf_bins % 2 == 1, "RDF bins must be odd for the Simpson's rule calculation.");
    auto rdf = freud::density::RDF(rdf_bins, r_max);
    rdf.accumulate(neighbor_query, query_points, n_query_points, nlist, qargs);

    auto const rdf_centers = rdf.getBinCenters()[0];
    auto const rdf_values = rdf.getRDF();
    auto const k_bin_centers = m_histogram.getBinCenters()[0];

    util::forLoopWrapper(0, k_bin_centers.size(), [&](size_t begin_k_index, size_t end_k_index) {
        for (size_t k_index = begin_k_index; k_index < end_k_index; ++k_index)
        {
            auto const k = k_bin_centers[k_index];

            auto integrand = [&](size_t rdf_index) {
                auto const r = rdf_centers[rdf_index];
                auto const g_r = rdf_values[rdf_index];
                return r * r * (g_r - 1.0) * util::sinc(k * r);
            };

            auto const dr
                = (rdf_centers.back() - rdf_centers.front()) / static_cast<float>(rdf_centers.size());
            auto const integral = util::simpson_integrate(integrand, rdf_bins, dr);
            m_local_histograms.increment(k_index, integral);
        }
    });
}

const util::ManagedArray<float>& StructureFactor::getStructureFactor()
{
    if (m_reduce)
    {
        m_local_histograms.reduceInto(m_structure_factor);
    }

    if (!m_direct)
    {
        util::forLoopWrapper(0, m_structure_factor.size(), [this](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                m_structure_factor[i] = 1.0 + m_normalization * m_structure_factor[i];
            }
        });
    }
    return m_structure_factor;
}

}; }; // namespace freud::diffraction
