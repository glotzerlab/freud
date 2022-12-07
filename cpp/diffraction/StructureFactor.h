#ifndef STRUCTURE_FACTOR_H
#define STRUCTURE_FACTOR_H

#include <limits>
#include <vector>

#include "Histogram.h"

namespace freud { namespace diffraction {

/* Abstract base class for all 1D structure factors
 *
 * Derived structure factors may be either static or time-dependent. They may
 * sample vectors in reciprocal space, or they may just use the magnitude of the
 * k-vectors. All structure factor calculations require an x-axis which is the
 * magnitude of the k-vector, however because some methods compute structure
 * factors in a time-dependent fashion (like the IntermediateScattering function),
 * this class does not contain a member variable related to the x-axis.
 *
 * All structure factors will need to get the range of k-values the x-axis is
 * plotted on, and the minimum value of k for which the calculation is valid.
 *
 * */
class StructureFactor
{
public:
    StructureFactor(unsigned int bins, float k_max, float k_min = 0)
        : m_nbins(bins), m_k_max(k_max), m_k_min(k_min)
    {}

    virtual ~StructureFactor() = default;

    //<! Get the centers of the k-vector bins
    virtual std::vector<float> getBinCenters() const = 0;

    //<! Get the edges of the k-vector bins
    virtual std::vector<float> getBinEdges() const = 0;

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

    //!< Get the maxmimum k-vector magnitude
    float getKMax() const
    {
        return m_k_max;
    }

    //!< Get the minimum k-vector magnitude
    float getKMin() const
    {
        return m_k_min;
    }

protected:
    //<! alias for the histogram class used by inheriting classes
    using StructureFactorHistogram = util::Histogram<float>;

    //!< minimum k-vector magnitude for which the calculation is valid
    float m_min_valid_k {std::numeric_limits<float>::infinity()};

    //!< number of k values between k_min and k_max to use
    unsigned int m_nbins;

    //!< maximum and minimum k values
    float m_k_max;
    float m_k_min;
};

}; }; // namespace freud::diffraction

#endif // STRUCTURE_FACTOR_H
