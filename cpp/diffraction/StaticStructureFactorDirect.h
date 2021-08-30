// Copyright (c) 2010-2020 The Regents of the University of Michigan
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

class StaticStructureFactorDirect
{
    using StructureFactorHistogram = util::Histogram<float>;
    using KBinHistogram = util::Histogram<unsigned int>;

public:
    //! Constructor
    StaticStructureFactorDirect(unsigned int bins, float k_max, float k_min = 0, unsigned int max_k_points = 10000);

    //! Destructor
    virtual ~StaticStructureFactorDirect() = default;

    //! Compute the structure factor S(k) using the direct formula
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points, unsigned int n_total);

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
        m_local_structure_factor.reset();
        m_local_histograms.reset();
        m_min_valid_k = std::numeric_limits<float>::infinity();
        m_reduce = true;
    }

    //! Get the structure factor
    const util::ManagedArray<float>& getStructureFactor()
    {
        return reduceAndReturn(m_structure_factor.getBinCounts());
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

private:
    //! Compute the complex amplitude F(k) for a set of points and k points
    static std::vector<std::complex<float>> compute_F_k(const vec3<float>* points, unsigned int n_points,
                                                        unsigned int n_total, const vec3<float>* k_points,
                                                        unsigned int n_k_points);

    //! Compute the static structure factor S(k) for all k points
    static std::vector<float> compute_S_k(const std::vector<std::complex<float>>& F_k_points,
                                          const std::vector<std::complex<float>>& F_k_query_points);

    void reciprocal_isotropic(const box::Box& box);

    unsigned int m_max_k_points;                 //!< Target number of k-vectors to sample
    std::vector<vec3<float>> m_k_vectors;        //!< k-vectors used for sampling
    StructureFactorHistogram m_structure_factor; //!< Histogram to hold computed structure factor
    StructureFactorHistogram::ThreadLocalHistogram
        m_local_structure_factor; //!< Thread local histograms for TBB parallelism
    KBinHistogram m_histogram;    //!< Histogram of sampled k bins
    KBinHistogram::ThreadLocalHistogram
        m_local_histograms; //!< Thread local histograms of sampled k bins for TBB parallelism
    float m_min_valid_k {
        std::numeric_limits<float>::infinity()}; //!< The minimum valid k-value based on the computed box
    bool m_reduce {true};                        //!< Whether to reduce
};

std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min = 0,
                                              unsigned int max_k_points = 10000);

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_DIRECT_H
