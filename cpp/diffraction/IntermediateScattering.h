// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef INTERMEDIATE_SCATTERING_H
#define INTERMEDIATE_SCATTERING_H

#include <complex>
#include <limits>
#include <vector>

#include "Box.h"
#include "Histogram.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "StaticStructureFactorDirect.h"

/*! \file IntermediateScattering.h
    \brief Routines for computing intermediate scattering function.

    This computes the intermediate scattering function F(k, t) between sets of
    time-dependent points and query points.

    The k-vectors are used to compute complex scattering amplitudes F(k, t) by
    summing over the scattering contribution of all particle positions (atomic
    form factors are assumed to be 1) at every k-vector. The complex conjugate
    of the scattering amplitudes of the points at each k-vector are multiplied
    by the scattering amplitudes of the query points at each k-vector. Finally,
    the results are binned according to their k-vector's magnitude and normalized
    by the number of samples in each radial bin in k-space.

    Note that k-vectors are in the physics convention, and q-vectors are in the
    crystallographic convention. These conventions differ by a factor of 2\pi.

    See also:
    https://en.wikipedia.org/wiki/Reciprocal_lattice#Arbitrary_collection_of_atoms
*/

namespace freud { namespace diffraction {

class IntermediateScattering : public StructureFactorDirect
{
public:
    //! Constructor
    IntermediateScattering(const box::Box& box, unsigned int bins, float k_max, float k_min = 0,
                           unsigned int num_sampled_k_points = 0);

    void compute(const vec3<float>* points, unsigned int num_points, const vec3<float>* query_points,
                 unsigned int num_query_points, unsigned int num_frames, unsigned int n_total);

    const util::ManagedArray<float>& getSelfFunction()
    {
        return m_structure_factor.getBinCounts();
    }

    const util::ManagedArray<float>& getDistinctFunction()
    {
        return reduceAndReturn(m_structure_factor_distinct.getBinCounts());
    }

private:
    //!< box for the calculation, we assume the box is constant over the time interval
    box::Box m_box;

    //!< Histogram to hold computed structure factor
    StructureFactorHistogram m_structure_factor_distinct;
    //!< Thread local histograms for TBB parallelism
    StructureFactorHistogram::ThreadLocalHistogram m_local_structure_factor_distinct;

    //!< Histogram of sampled k bins, used to normalize S(q)
    KBinHistogram m_k_histogram_distinct;
    //!< Thread local histograms of sampled k bins for TBB parallelism
    KBinHistogram::ThreadLocalHistogram m_local_k_histograms_distinct;

    //!< Helpers to compute self and distinct parts
    std::vector<std::complex<float>> compute_self(const vec3<float>* rt, const vec3<float>* r0,
                                                  unsigned int n_points, unsigned int n_total,
                                                  const std::vector<vec3<float>>& k_points);

    std::vector<std::complex<float>> compute_distinct(const vec3<float>* rt, const vec3<float>* r0,
                                                      unsigned int n_points, unsigned int n_total,
                                                      const std::vector<vec3<float>>& k_points);
};

}; }; // namespace freud::diffraction

#endif // INTERMEDIATE_SCATTERING_H
