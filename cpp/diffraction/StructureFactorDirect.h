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
    //!< Protected constructor makes the class abstract
    StructureFactorDirect(unsigned int bins, float k_max, float k_min = 0,
        unsigned int num_sampled_k_points = 0) : StructureFactor(bins, k_max, k_min),
        m_num_sampled_k_points(num_sampled_k_points) {}

    //! Sample reciprocal space isotropically to get k points
    static std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min,
                                                         unsigned int num_sampled_k_points);

    //!< Number of k points to use in the calculation
    unsigned int m_num_sampled_k_points;

    //!< the k points used in the calculation
    std::vector<vec3<float>> m_k_points;
};

};}; // namespace freud::diffraction

#endif // STRUCTURE_FACTOR_DIRECT_H
