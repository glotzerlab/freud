// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_DEBYE_H
#define STATIC_STRUCTURE_FACTOR_DEBYE_H

#include "Box.h"
#include "Histogram.h"
#include "NeighborQuery.h"

/*! \file StaticStructureFactorDirect.h
    \brief Routines for computing static structure factors.

    Computes structure factors from the Fourier transform of a radial distribution function (RDF).
    This method is not capable of resolving k-vectors smaller than the magnitude 4 * pi / L, where L is the
    smallest side length of the system's periodic box.
*/

namespace freud { namespace diffraction {

class StaticStructureFactorDirect
{
    using StructureFactorHistogram = util::Histogram<float>;
    using KBinHistogram = util::Histogram<unsigned int>;

public:
    //! Constructor
    StaticStructureFactorDirect(unsigned int bins, float k_max, float k_min = 0);

    //! Destructor
    virtual ~StaticStructureFactorDirect() = default;

    //! Compute the structure factor S(k) using the direct formula
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points, unsigned int n_total, const vec3<float>* k_points, unsigned int n_k_points);

    //! Reduce thread-local arrays onto the primary data arrays.
    void reduce();

    //! Return thing_to_return after reducing if necessary.
    template<typename U> U& reduceAndReturn(U& thing_to_return)
    {
        if (m_reduce)
        {
            reduce();
        }
        m_reduce = false;
        return thing_to_return;
    }

    //! Reset the histogram to all zeros
    void reset()
    {
        m_local_histograms.reset();
        m_k_bin_local_histograms.reset();
        m_frame_counter = 0;
        m_reduce = true;
    }

    //! Get the structure factor
    const util::ManagedArray<float>& getStructureFactor()
    {
        return reduceAndReturn(m_structure_factor);
    }

    //! Get the k bin edges
    std::vector<float> getBinEdges() const
    {
        return m_histogram.getBinEdges()[0];
    }

    //! Get the k bin centers
    std::vector<float> getBinCenters() const
    {
        return m_histogram.getBinCenters()[0];
    }

    //! Get the minimum valid k value
    float getMinValidK() const
    {
        return m_min_valid_k;
    }

private:
    //! Compute the complex amplitude F(k) for a set of points and k points
    static std::vector<std::complex<float>> compute_F_k(const vec3<float>* points, unsigned int n_points,
            unsigned int n_total, const vec3<float>* k_points, unsigned int n_k_points);

    //! Compute the static structure factor S(k) for all k points
    static std::vector<float> compute_S_k(const std::vector<std::complex<float>>& F_k_points,
            const std::vector<std::complex<float>>& F_k_query_points);

    StructureFactorHistogram m_histogram; //!< Histogram to hold computed structure factor
    StructureFactorHistogram::ThreadLocalHistogram m_local_histograms; //!< Thread local histograms for TBB parallelism
    KBinHistogram m_k_bin_histogram; //!< Histogram of sampled k bins
    KBinHistogram::ThreadLocalHistogram m_k_bin_local_histograms; //!< Thread local histograms of sampled k bins for TBB parallelism
    util::ManagedArray<float> m_structure_factor; //!< The computed structure factor
    unsigned int m_frame_counter {0};                 //!< Number of frames calculated
    float m_min_valid_k {std::numeric_limits<float>::infinity()}; //!< The minimum valid k-value based on the computed box
    bool m_reduce {true};                                //!< Whether to reduce
};

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_DIRECT_H
