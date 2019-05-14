// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_DESCRIPTORS_H
#define LOCAL_DESCRIPTORS_H

#include <memory>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "fsph/src/spherical_harmonics.hpp"

#include "tbb/atomic.h"

/*! \file LocalDescriptors.h
  \brief Computes local descriptors.
*/

namespace freud { namespace environment {

enum LocalDescriptorOrientation {
    LocalNeighborhood,
    Global,
    ParticleLocal};

/*! Compute a set of descriptors (a numerical "fingerprint") of a
*  particle's local environment.
*/
class LocalDescriptors
    {
public:
    //! Constructor
    //!
    //! \param neighmax Maximum number of neighbors to compute descriptors for
    //! \param lmax Maximum spherical harmonic l to consider
    //! \param rmax Initial guess of the maximum radius to look for n_neigh neighbors
    //! \param negative_m whether to calculate Ylm for negative m
    LocalDescriptors(unsigned int neighmax,
                     unsigned int lmax, float rmax, bool negative_m);

    //! Get the maximum number of neighbors
    unsigned int getNeighmax() const
        {
        return m_neighmax;
        }

    //! Get the last number of spherical harmonics computed
    unsigned int getNSphs() const
        {
        return m_nSphs;
        }

    //! Get the maximum spherical harmonic l to calculate for
    unsigned int getLMax() const
        {
        return m_lmax;
        }

    //! Get the number of particles
    unsigned int getNP() const
        {
        return m_Nref;
        }

    //! Compute the nearest neighbors for each particle
    void computeNList(const box::Box& box, const vec3<float> *r_ref,
                      unsigned int Nref, const vec3<float> *r, unsigned int Np);

    //! Compute the local neighborhood descriptors given some
    //! positions and the number of particles
    void compute(const box::Box& box,
                 const freud::locality::NeighborList *nlist,
                 unsigned int nNeigh,
                 const vec3<float> *r_ref, unsigned int Nref,
                 const vec3<float> *r, unsigned int Np,
                 const quat<float> *q_ref,
                 LocalDescriptorOrientation orientation);

    //! Get a reference to the last computed spherical harmonic array
    std::shared_ptr<std::complex<float> > getSph()
        {
        return m_sphArray;
        }

    unsigned int getSphWidth() const
        {
        return fsph::sphCount(m_lmax) +
            (m_lmax > 0 && m_negative_m ? fsph::sphCount(m_lmax - 1): 0);
        }

private:
    unsigned int m_neighmax;          //!< Maximum number of neighbors to calculate
    unsigned int m_lmax;              //!< Maximum spherical harmonic l to calculate
    bool m_negative_m;                //!< true if we should compute Ylm for negative m
    locality::NearestNeighbors m_nn;  //!< NearestNeighbors to find neighbors with
    unsigned int m_Nref;              //!< Last number of points computed
    unsigned int m_nSphs;             //!< Last number of bond spherical harmonics computed

    //! Spherical harmonics for each neighbor
    std::shared_ptr<std::complex<float> > m_sphArray;
    };

}; }; // end namespace freud::environment

#endif // LOCAL_DESCRIPTORS_H
