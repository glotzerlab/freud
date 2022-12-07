// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_H
#define STATIC_STRUCTURE_FACTOR_H

#include <limits>
#include <vector>

#include "StructureFactor.h"
#include "Histogram.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"

/*! \file StaticStructureFactor.h
    \brief Base class for static tructure factor classes.
*/

namespace freud { namespace diffraction {

/* Abstract base class for all static structure factors.
 *
 * A static structure factor is a structure factor which can be computed using
 * only the data from one frame of a simulation. A typical use case is to compute
 * the static structure factor for each frame over many frames of a simulation,
 * and get an average for better statistics/curve smoothness. To support this use
 * case, all static structure factors must have logic for either continuing to
 * accumulate histogram data or resetting the data on each successive call to
 * accumulate().
 *
 * */
class StaticStructureFactor : virtual public StructureFactor
{
protected:
    StaticStructureFactor(unsigned int bins, float k_max, float k_min = 0);

public:
    virtual ~StaticStructureFactor() = default;

    virtual void accumulate(const freud::locality::NeighborQuery* neighbor_query,
                            const vec3<float>* query_points, unsigned int n_query_points,
                            unsigned int n_total)
        = 0;

    virtual void reset() = 0;

    //! Get the structure factor
    const util::ManagedArray<float>& getStructureFactor()
    {
        return reduceAndReturn(m_structure_factor.getBinCounts());
    }

    //! Get the k bin edges
    std::vector<float> getBinEdges() const override
    {
        return m_structure_factor.getBinEdges()[0];
    }

    //! Get the k bin centers
    std::vector<float> getBinCenters() const override
    {
        return m_structure_factor.getBinCenters()[0];
    }

protected:
    virtual void reduce() = 0;

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

    bool m_reduce {true}; //! Whether to reduce local histograms
};

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_H
