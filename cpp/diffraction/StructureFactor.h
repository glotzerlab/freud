#ifndef STRUCTURE_FACTOR_H
#define STRUCTURE_FACTOR_H

#include <limits>
#include <vector>

#include "Histogram.h"

namespace freud { namespace diffraction {

/* Abstract base class for all structure factors
 *
 * Derived structure factors may be either static or time-dependent. They may
 * sample vectors in reciprocal space, or they may just use the magnitude of the
 * k-vectors.
 *
 * By assumption, all structure factor calculations will compute the structure
 * factor histogram in parallel, with each thread doing an individual calculation,
 * then the local histograms will be reduced into a single result in the reduce
 * method.
 * */
class StructureFactor
{
public:
    StructureFactor(unsigned int bins, float k_max, float k_min = 0)
        : m_nbins(bins), m_k_min(k_min), m_k_max(k_max)
    {}

    virtual ~StructureFactor() = default;

    //! Get the k bin edges
    virtual std::vector<float> getBinEdges() const = 0;

    //! Get the k bin centers
    virtual std::vector<float> getBinCenters() const = 0;

    //<! Get the minimum valid value of k for which the calculation is valid
    float getMinValidK() const
    {
        return m_min_valid_k;
    }

    //!< Get the number of k-vector magnitudes for the calculation
    unsigned int getNumBins() const
    {
        return m_nbins;
    }

protected:
    //<! alias for the histogram class used by inheriting classes
    using StructureFactorHistogram = util::Histogram<float>;

    //!< minimum k-vector magnitude for which the calculation is valid
    float m_min_valid_k {std::numeric_limits<float>::infinity()};

    //!< number of k values between k_min and k_max to use
    unsigned int m_nbins;

    //!< max and min k values
    float m_k_min;
    float m_k_max;

    //!< reduce thread-local histograms into a single histogram
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

    bool m_reduce {true}; //! Whether to reduce local histograms
};

}; }; // namespace freud::diffraction

#endif // STRUCTURE_FACTOR_H
