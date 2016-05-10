#include <boost/shared_array.hpp>

#include "LinkCell.h"
// hack to keep VectorMath's swap from polluting the global namespace
#include "VectorMath.h"
#include "trajectory.h"

#include "tbb/atomic.h"

#include "fsph/src/spherical_harmonics.hpp"

#ifndef _LOCAL_DESCRIPTORS_H__
#define _LOCAL_DESCRIPTORS_H__

/*! \file LocalDescriptors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

/*! Compute a set of descriptors (a numerical "fingerprint") of a
*  particle's local environment.
*/
class LocalDescriptors
    {
public:
    //! Constructor
    //!
    //! \param box This frame's box
    //! \param nNeigh Number of neighbors to compute descriptors for
    //! \param lmax Maximum spherical harmonic l to consider
    //! \param rmax Initial guess of the maximum radius to look for n_neigh neighbors
    //! \param negative_m whether to calculate Ylm for negative m
    LocalDescriptors(const trajectory::Box& box, unsigned int nNeigh,
                     unsigned int lmax, float rmax, bool negative_m);

    //! Get the simulation box
    const trajectory::Box& getBox() const
        {
        return m_box;
        }

    //! Get the number of neighbors
    unsigned int getNNeigh() const
        {
        return m_nNeigh;
        }

    //! Get the maximum spherical harmonic l to calculate for
    unsigned int getLMax() const
        {
        return m_lmax;
        }

    //! Get the current cutoff radius used
    float getRMax() const
        {
        return m_rmax;
        }

    //! Get the number of particles
    unsigned int getNP() const
        {
        return m_Np;
        }

    //! Compute the local neighborhood descriptors given some
    //! positions, orientations, and the number of particles
    void compute(const vec3<float> *r, const quat<float> *q, unsigned int Np);

    // //! Python wrapper for compute
    // void computePy(boost::python::numeric::array r,
    //     boost::python::numeric::array q);

    //! Get a reference to the last computed radius magnitude array
    boost::shared_array<float> getMagR()
        {
        return m_magrArray;
        }

    //! Get a reference to the last computed relative orientation array
    boost::shared_array<quat<float> > getQij()
        {
        return m_qijArray;
        }

    //! Get a reference to the last computed spherical harmonic array
    boost::shared_array<std::complex<float> > getSph()
        {
        return m_sphArray;
        }

    unsigned int getSphWidth() const
        {
        return fsph::sphCount(m_lmax) +
            (m_lmax > 0 && m_negative_m ? fsph::sphCount(m_lmax - 1): 0);
        }

    // //! Python wrapper for getMagR() (returns a copy)
    // boost::python::numeric::array getMagRPy()
    //     {
    //     const intp cshape[] = {m_Np, m_nNeigh};
    //     const std::vector<intp> shape(cshape, cshape + sizeof(cshape)/sizeof(intp));
    //     float *arr = m_magrArray.get();
    //     return num_util::makeNum(arr, shape);
    //     }

    // //! Python wrapper for getQij() (returns a copy)
    // boost::python::numeric::array getQijPy()
    //     {
    //     const intp cshape[] = {m_Np, m_nNeigh, 4};
    //     const std::vector<intp> shape(cshape, cshape + sizeof(cshape)/sizeof(intp));
    //     float *arr = (float*) m_qijArray.get();
    //     return num_util::makeNum(arr, shape);
    //     }

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
    trajectory::Box m_box;            //!< Simulation box the particles belong in
    unsigned int m_nNeigh;            //!< Number of neighbors to calculate
    unsigned int m_lmax;              //!< Maximum spherical harmonic l to calculate
    float m_rmax;                     //!< Maximum r at which to determine neighbors
    bool m_negative_m;                //!< true if we should compute Ylm for negative m
    locality::LinkCell m_lc;          //!< LinkCell to bin particles for the computation
    unsigned int m_Np;                //!< Last number of points computed
    tbb::atomic<unsigned int> m_deficits; //!< Neighbor deficit count from the last compute step

    //! Magnitude of the radius vector for each neighbor
    boost::shared_array<float> m_magrArray;
    //! Quaternion to rotate into each neighbor's orientation
    boost::shared_array<quat<float> > m_qijArray;
    //! Spherical harmonics for each neighbor
    boost::shared_array<std::complex<float> > m_sphArray;
    };

}; }; // end namespace freud::order

#endif // _LOCAL_DESCRIPTORS_H__
