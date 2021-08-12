// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_UTIL_H
#define STATIC_STRUCTURE_FACTOR_UTIL_H

#include <complex>

#include "VectorMath.h"

/*! \file StaticStructureFactorUtil.h
    \brief Routines for computing static structure factors.

    Computes complex scattering amplitudes F(k) for a set of k-vectors by
    summing over the scattering contribution of all particle positions (atomic
    form factors are assumed to be 1). Note that the k-vectors are in the
    physics convention (not crystallographic convention), so no factor of 2\pi
    is needed in the formula. See also:
    https://en.wikipedia.org/wiki/Reciprocal_lattice#Arbitrary_collection_of_atoms
*/

namespace freud { namespace diffraction {

void compute_F_k(const vec3<float>* points, const unsigned int n_points,
        const vec3<float>* k_points, const unsigned int n_k_points, std::complex<float>* F_k);

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_UTIL_H
