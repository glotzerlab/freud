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

    This computes the intermediate scattering function F(k, t) between sets of time-dependent points and
    query points. First, k-vectors are sampled isotropically, with each radial
    bin containing an equal density of k points. Next, the k-vectors are used to
    compute complex scattering amplitudes F(k, t) by summing over the scattering
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

class IntermediateScattering : public StaticStructureFactorDirect
{
    using KBinHistogram = util::Histogram<unsigned int>;

public:
    //! Reset the histogram to all zeros
    void reset() override
    {
        m_local_structure_factor.reset();
        m_local_k_histograms.reset();
        m_min_valid_k = std::numeric_limits<float>::infinity();
        m_reduce = true;
        box_assigned = false;
    }

private:
    // to record the position r0 of the first frame
    bool m_first_call {true};
    static std::vector<vec3<float>> m_r0;
};

}; }; // namespace freud::diffraction

#endif // INTERMEDIATE_SCATTERING_H
