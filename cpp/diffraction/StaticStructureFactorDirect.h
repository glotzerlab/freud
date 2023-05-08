// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_DIRECT_H
#define STATIC_STRUCTURE_FACTOR_DIRECT_H

#include <complex>
#include <limits>
#include <vector>

#include "Box.h"
#include "Histogram.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "StaticStructureFactor.h"

/*! \file StaticStructureFactorDirect.h
    \brief Routines for computing static structure factors.

    This computes the static structure factor S(k) between sets of points and
    query points. First, k-vectors are sampled isotropically, with each radial
    bin containing an equal density of k points. Next, the k-vectors are used to
    compute complex scattering amplitudes F(k) by summing over the scattering
    contribution of all particle positions (atomic form factors are assumed to
    be 1) at every k-vector. The complex conjugate of the scattering amplitudes
    of the points at each k-vector are multiplied by the scattering amplitudes
    of the query points at each k-vector. Finally, the results are binned
    according to their k-vector's magnitude and normalized by the number of
    samples in each radial bin in k-space.

    Note that k-vectors are in the physics convention, and q-vectors are in the
    crystallographic convention. These conventions differ by a factor of 2\pi.

    The methods in this class are based on algorithms in the MIT-licensed
    dynasor package available here:
    https://dynasor.materialsmodeling.org/

    See also:
    https://en.wikipedia.org/wiki/Reciprocal_lattice#Arbitrary_collection_of_atoms
*/

namespace freud { namespace diffraction {

class StaticStructureFactorDirect : public StaticStructureFactor
{
    using KBinHistogram = util::Histogram<unsigned int>;

public:
    //! Constructor
    StaticStructureFactorDirect(unsigned int bins, float k_max, float k_min = 0,
                                unsigned int num_sampled_k_points = 0);

    //! Compute the structure factor S(k) using the direct formula
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points, unsigned int n_total) override;

    //! Reset the histogram to all zeros
    void reset() override
    {
        m_local_structure_factor.reset();
        m_local_k_histograms.reset();
        m_min_valid_k = std::numeric_limits<float>::infinity();
        m_reduce = true;
        box_assigned = false;
    }

    //! Get the number of sampled k points
    unsigned int getNumSampledKPoints() const
    {
        return m_num_sampled_k_points;
    }

    //! Get the k points last used
    std::vector<vec3<float>> getKPoints() const
    {
        return m_k_points;
    }

private:
    //! Reduce thread-local arrays onto the primary data arrays.
    void reduce() override;

    //! Compute the complex amplitude F(k) for a set of points and k points
    static std::vector<std::complex<float>> compute_F_k(const vec3<float>* points, unsigned int n_points,
                                                        unsigned int n_total,
                                                        const std::vector<vec3<float>>& k_points);

    //! Compute the static structure factor S(k) for all k points
    static std::vector<float> compute_S_k(const std::vector<std::complex<float>>& F_k_points,
                                          const std::vector<std::complex<float>>& F_k_query_points);

    //! Sample reciprocal space isotropically to get k points
    static std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min,
                                                         unsigned int num_sampled_k_points);

    unsigned int m_num_sampled_k_points; //!< Target number of k-vectors to sample
    std::vector<vec3<float>> m_k_points; //!< k-vectors used for sampling
    KBinHistogram m_k_histogram;         //!< Histogram of sampled k bins, used to normalize S(q)
    KBinHistogram::ThreadLocalHistogram
        m_local_k_histograms;  //!< Thread local histograms of sampled k bins for TBB parallelism
    box::Box previous_box;     //!< box assigned to the system
    bool box_assigned {false}; //!< Whether to reuse the box
};

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_DIRECT_H
