
#ifndef DIFFRACTION_PATTERN_H
#define DIFFRACTION_PATTERN_H

#include <complex>
#include <vector>

#include "VectorMath.h"
//#include "Box.h"
//#include "Histogram.h"
//#include "ManagedArray.h"
//#include "NeighborQuery.h"

namespace freud { namespace diffraction {

class DiffractionPattern
{
//    using KBinHistogram = util::Histogram<unsigned int>;

public:
//    //! Constructor
//    DiffractionPattern(unsigned int bins, float k_max, float k_min = 0,
//                                unsigned int num_sampled_k_points = 0);

//    //! Compute the structure factor S(k) using the direct formula
//    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
//                    unsigned int n_query_points, unsigned int n_total) override;
//
//    //! Reset the histogram to all zeros
//    void reset() override
//    {
//        m_local_structure_factor.reset();
//        m_local_k_histograms.reset();
//        m_min_valid_k = std::numeric_limits<float>::infinity();
//        m_reduce = true;
//        box_assigned = false;
//    }
//
//    //! Get the number of sampled k points
//    unsigned int getNumSampledKPoints() const
//    {
//        return m_num_sampled_k_points;
//    }
//
//    //! Get the k points last used
//    std::vector<vec3<float>> getKPoints() const
//    {
//        return m_k_points;
//    }

    //! Compute the complex amplitude F(k) for a set of points and k points
    static std::vector<std::complex<float>> compute_F_k(const vec3<float>* points, unsigned int n_points,
                                                        unsigned int n_total,
                                                        const std::vector<vec3<float>>& k_points);

    //! Compute the static structure factor S(k) for all k points
    static std::vector<float> compute_S_k(const std::vector<std::complex<float>>& F_k_points,
                                          const std::vector<std::complex<float>>& F_k_query_points);

//private:
//    //! Reduce thread-local arrays onto the primary data arrays.
//    void reduce() override;
//
//    //! Sample reciprocal space isotropically to get k points
//    static std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min,
//                                                         unsigned int num_sampled_k_points);
//
//    unsigned int m_num_sampled_k_points; //!< Target number of k-vectors to sample
//    std::vector<vec3<float>> m_k_points; //!< k-vectors used for sampling
//    KBinHistogram m_k_histogram;         //!< Histogram of sampled k bins, used to normalize S(q)
//    KBinHistogram::ThreadLocalHistogram
//        m_local_k_histograms;  //!< Thread local histograms of sampled k bins for TBB parallelism
//    box::Box previous_box;     //!< box assigned to the system
//    bool box_assigned {false}; //!< Whether to reuse the box
};

}; }; // namespace freud::diffraction

#endif // DIFFRACTION_PATTERN_H
