// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_H
#define STATIC_STRUCTURE_FACTOR_H

#include <limits>
#include <vector>

#include "Histogram.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"

/*! \file StaticStructureFactor.h
    \brief Base class for structure factor classes.
*/

namespace freud { namespace diffraction {

class StaticStructureFactor
{
protected:
    using StructureFactorHistogram = util::Histogram<float>;

    StaticStructureFactor(unsigned int bins, float k_max, float k_min = 0);

public:
    virtual ~StaticStructureFactor() = default;

    virtual void accumulate(std::shared_ptr<locality::NeighborQuery> neighbor_query,
                            const vec3<float>* query_points, unsigned int n_query_points,
                            unsigned int n_total) {}; // Note: this should be pure

    virtual void reset() {}; // Note: this should be pure

    //! Get the structure factor
    std::shared_ptr<util::ManagedArray<float>> getStructureFactor()
    {
        // reduceAndReturn requires a reference, but can't get it from an unnamed temporary variable
        auto bin_counts = m_structure_factor.getBinCounts();

        return reduceAndReturn(bin_counts);
    }

    //! Get the k bin edges
    std::vector<float> getBinEdges() const
    {
        return m_structure_factor.getBinEdges()[0];
    }

    //! Get the k bin centers
    std::vector<float> getBinCenters() const
    {
        return m_structure_factor.getBinCenters()[0];
    }

    //! Get the minimum valid k value
    float getMinValidK() const
    {
        return m_min_valid_k;
    }

protected:
    virtual void reduce() {};

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

    StructureFactorHistogram m_structure_factor; //!< Histogram to hold computed structure factor
    StructureFactorHistogram::ThreadLocalHistogram
        m_local_structure_factor; //!< Thread local histograms for TBB parallelism

    bool m_reduce {true};                                         //! Whether to reduce local histograms
    float m_min_valid_k {std::numeric_limits<float>::infinity()}; //! Minimum valid k-vector magnitude
};

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_H
