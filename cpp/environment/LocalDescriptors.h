// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_DESCRIPTORS_H
#define LOCAL_DESCRIPTORS_H

#include <complex>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"

/*! \file LocalDescriptors.h
  \brief Computes local descriptors.
*/

namespace freud { namespace environment {

enum LocalDescriptorOrientation
{
    LocalNeighborhood,
    Global,
    ParticleLocal
};

/*! Compute a set of descriptors (a numerical "fingerprint") of a
 *  particle's local environment.
 */
class LocalDescriptors
{
public:
    //! Constructor
    //!
    //! \param l_max Maximum spherical harmonic l to consider
    //! \param negative_m whether to calculate Ylm for negative m
    LocalDescriptors(unsigned int l_max, bool negative_m, LocalDescriptorOrientation orientation);

    //! Get the last number of spherical harmonics computed
    unsigned int getNSphs() const
    {
        return m_nSphs;
    }

    //! Get the maximum spherical harmonic l to calculate for
    unsigned int getLMax() const
    {
        return m_l_max;
    }

    //! Compute the local neighborhood descriptors given some
    //! positions and the number of particles
    void compute(const locality::NeighborQuery* nq, const vec3<float>* query_points,
                 unsigned int n_query_points, const quat<float>* orientations,
                 const freud::locality::NeighborList* nlist, locality::QueryArgs qargs,
                 unsigned int max_num_neighbors = 0);

    //! Get a reference to the last computed spherical harmonic array
    const util::ManagedArray<std::complex<float>>& getSph() const
    {
        return m_sphArray;
    }

    //! Return the number of spherical harmonics that will be computed for each bond.
    unsigned int getSphWidth() const
    {
        return fsph::sphCount(m_l_max) + (m_l_max > 0 && m_negative_m ? fsph::sphCount(m_l_max - 1) : 0);
    }

    //! Return a pointer to the NeighborList used in the last call to compute.
    locality::NeighborList* getNList()
    {
        return &m_nlist;
    }

    bool getNegativeM() const
    {
        return m_negative_m;
    }

    LocalDescriptorOrientation getMode() const
    {
        return m_orientation;
    }

private:
    unsigned int m_l_max;                     //!< Maximum spherical harmonic l to calculate
    bool m_negative_m;                        //!< true if we should compute Ylm for negative m
    unsigned int m_nSphs;                     //!< Last number of bond spherical harmonics computed
    locality::NeighborList m_nlist;           //!< The NeighborList used in the last call to compute.
    LocalDescriptorOrientation m_orientation; //!< The orientation mode to compute with.

    //! Spherical harmonics for each neighbor
    util::ManagedArray<std::complex<float>> m_sphArray;
};

}; }; // end namespace freud::environment

#endif // LOCAL_DESCRIPTORS_H
