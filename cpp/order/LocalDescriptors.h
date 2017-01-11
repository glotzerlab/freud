// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>

#include "NearestNeighbors.h"
// hack to keep VectorMath's swap from polluting the global namespace
#include "VectorMath.h"
#include "box.h"

#include "tbb/atomic.h"

#include "../../extern/fsph/src/spherical_harmonics.hpp"

#ifndef _LOCAL_DESCRIPTORS_H__
#define _LOCAL_DESCRIPTORS_H__

/*! \file LocalDescriptors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

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

    //! Get the last number of neighbors
    unsigned int getNNeigh() const
        {
        return m_nNeigh;
        }

    //! Get the maximum spherical harmonic l to calculate for
    unsigned int getLMax() const
        {
        return m_lmax;
        }

    //! Get the maximum neighbor distance
    unsigned int getRMax() const
        {
        return m_nn.getRMax();
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
    void compute(const box::Box& box, unsigned int nNeigh,
                 const vec3<float> *r_ref, unsigned int Nref,
                 const vec3<float> *r, unsigned int Np,
                 const quat<float> *q_ref,
                 LocalDescriptorOrientation orientation);

    // //! Python wrapper for compute
    // void computePy(boost::python::numeric::array r,
    //     boost::python::numeric::array q);

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

    // //! Python wrapper for getSph() (returns a copy)
    // boost::python::numeric::array getSphPy()
    //     {
    //     // we have lmax**2 + 2*lmax + 1 spherical harmonics per
    //     // neighbor, but we don't keep Y00, so we have lmax**2 +
    //     // 2*lmax in total.
    //     const intp cshape[] = {m_Np, m_nNeigh, m_lmax*m_lmax + 2*m_lmax};
    //     const std::vector<intp> shape(cshape, cshape + sizeof(cshape)/sizeof(intp));
    //     std::complex<float> *arr = m_sphArray.get();
    //     return num_util::makeNum(arr, shape);
    //     }

private:
    unsigned int m_neighmax;          //!< Maximum number of neighbors to calculate
    unsigned int m_lmax;              //!< Maximum spherical harmonic l to calculate
    bool m_negative_m;                //!< true if we should compute Ylm for negative m
    locality::NearestNeighbors m_nn;  //!< NearestNeighbors to find neighbors with
    unsigned int m_Nref;              //!< Last number of points computed
    unsigned int m_nNeigh;            //!< Last number of neighbors computed

    //! Spherical harmonics for each neighbor
    std::shared_ptr<std::complex<float> > m_sphArray;
    };

}; }; // end namespace freud::order

#endif // _LOCAL_DESCRIPTORS_H__
