#ifndef STRUCTURE_FACTOR_DIRECT_H
#define STRUCTURE_FACTOR_DIRECT_H

#include "StructureFactor.h"

namespace freud { namespace diffraction {

/* Abstract base class for structure factors which sample k-space via the direct
 * method.
 *
 * The direct method of computing structure factors involves sampling a set of
 * points in k-space according to an isotropic distribution, where each radial
 * bin is sampled with an equal density. The algorithm to do this sampling is
 * taken from the MIT-licensed dyansor package located here:
 * https://dynasor.materialsmodeling.org/
 *
 * See also:
 * https://en.wikipedia.org/wiki/Reciprocal_lattice#Arbitrary_collection_of_atoms
 * */
class StructureFactorDirect : virtual public StructureFactor
{
public:
    //<! Get the number of k points used in the calculation
    unsigned int getNumSampledKPoints() const
    {
        return m_num_sampled_k_points;
    }

    //! Get the k points last used
    std::vector<vec3<float>> getKPoints() const
    {
        return m_k_points;
    }

protected:
    //!< type alias for bins for k-point sampling
    using KBinHistogram = util::Histogram<unsigned int>;

    //!< Protected constructor makes the class abstract
    StructureFactorDirect(unsigned int bins, float k_max, float k_min = 0,
                          unsigned int num_sampled_k_points = 0)
        : StructureFactor(bins, k_max, k_min), m_num_sampled_k_points(num_sampled_k_points),
          m_k_histogram(KBinHistogram(m_structure_factor.getAxes())),
          m_local_k_histograms(KBinHistogram::ThreadLocalHistogram(m_k_histogram))
    {}

    //! Sample reciprocal space isotropically to get k points
    static std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min,
                                                         unsigned int num_sampled_k_points);

    //!< Number of k points to use in the calculation
    unsigned int m_num_sampled_k_points;

    //!< the k points used in the calculation
    std::vector<vec3<float>> m_k_points;

    //!< histogram to hold the number of sampled k points in each k bin
    KBinHistogram m_k_histogram;
    KBinHistogram::ThreadLocalHistogram m_local_k_histograms; // thread-local version
};

}; }; // namespace freud::diffraction

#endif // STRUCTURE_FACTOR_DIRECT_H
