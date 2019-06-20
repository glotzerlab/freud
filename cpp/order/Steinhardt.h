// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STEINHARDT_H
#define STEINHARDT_H

#include <complex>
#include <memory>
#include <tbb/tbb.h>

#include "Box.h"
#include "NeighborList.h"
#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"
#include "wigner3j.h"

/*! \file Steinhardt.h
    \brief Computes variants of Steinhardt order parameters.
*/

namespace freud { namespace order {

//! Compute the Steinhardt local rotationally invariant Ql or Wl order parameter for a set of points
/*!
 * Implements the rotationally invariant Ql or Wl order parameter described
 * by Steinhardt. For a particle i, we calculate the average Q_l by summing
 * the spherical harmonics between particle i and its neighbors j in a local
 * region:
 * \f$ \overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b}
 * Y_{lm}(\theta(\vec{r}_{ij}),\phi(\vec{r}_{ij})) \f$
 *
 * This is then combined in a rotationally invariant fashion to remove local
 * orientational order as follows:
 * \f$ Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }  \f$
 *
 * If the average flag is set, the order parameters averages over the second neighbor shell.
 * For a particle i, we calculate the average Q_l by summing the spherical
 * harmonics between particle i and its neighbors j and the neighbors k of
 * neighbor j in a local region.
 *
 * If the norm flag is set, the Ql value is normalized by the average Qlm value
 * for the system.
 *
 * If the flag Wl is set, the third-order invariant Wl order parameter will
 * be calculated. Wl can aid in distinguishing between FCC, HCP, and BCC.
 *
 * For more details see:
 * - PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
 * - Wolfgang Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
 */

class Steinhardt
{
public:
    //! Steinhardt Class Constructor
    /*! Constructor for Steinhardt analysis class.
     *  \param rmax Cutoff radius for running the local order parameter.
     *              Values near first minima of the rdf are recommended.
     *  \param l Spherical harmonic number l.
     *           Must be a positive number.
     *  \param rmin (optional) Lower bound for computing the local order parameter.
     *                         Allows looking at, for instance, only the second shell,
     *                         or some other arbitrary rdf region.
     */
    Steinhardt(float rmax, unsigned int l, float rmin = 0, bool average = false, bool Wl = false)
        : m_Np(0), m_rmax(rmax), m_l(l), m_rmin(rmin), m_average(average), m_Wl(Wl)
    {
        // Error Checking
        if (m_rmax < 0.0f || m_rmin < 0.0f)
            throw std::invalid_argument("Steinhardt requires rmin and rmax must be positive.");
        if (m_rmin >= m_rmax)
            throw std::invalid_argument("Steinhardt requires rmin must be less than rmax.");
        if (m_l < 2)
            throw std::invalid_argument("Steinhardt requires l must be two or greater.");
    }

    //! Empty destructor
    virtual ~Steinhardt() {};

    //! Get the number of particles used in the last compute
    unsigned int getNP()
    {
        return m_Np;
    }

    //! Get the last calculated order parameter Ql
    std::shared_ptr<float> getQl()
    {
        if (m_average)
        {
            return m_QliAve;
        }
        else
        {
            return m_Qli;
        }
    }

    //! Get the last calculated order parameter Wl
    std::shared_ptr<std::complex<float>> getWl()
    {
        return m_WliOrder;
    }

    //! Get whether the Wl flag was set
    bool getUseWl()
    {
        return m_Wl;
    }

    //! Get Wl norm
    std::complex<float> getNormWl()
    {
        return m_NormWl;
    }

    //! Get Ql norm
    float getNorm()
    {
        return m_Norm;
    }

    //! Compute the order parameter
    virtual void compute(const box::Box& box, const locality::NeighborList* nlist, const vec3<float>* points,
                         unsigned int Np);

private:
    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduce();

    //! Spherical harmonics calculation for Ylm filling a
    //  vector<complex<float> > with values for m = -l..l.
    virtual void computeYlm(const float theta, const float phi, std::vector<std::complex<float>>& Ylm);

    template<typename T> std::shared_ptr<T> makeArray(size_t size);

    //! Reallocates only the necessary arrays when the number of particles changes
    // unsigned int Np number of particles
    void reallocateArrays(unsigned int Np);

    //! Calculates the base Ql order parameter before further modifications
    // if any.
    void baseCompute(const box::Box& box, const locality::NeighborList* nlist, const vec3<float>* points);

    //! Calculates the neighbor average Ql order parameter
    void computeAve(const box::Box& box, const locality::NeighborList* nlist, const vec3<float>* points);

    //! Normalize the order parameter
    float normalize();

    //! Normalize the Wl order parameter
    std::complex<float> normalizeWl();

    //! Sum over Wigner 3j coefficients to compute third-order invariants
    //  Wl from second-order invariants Ql
    void aggregateWl(std::shared_ptr<std::complex<float>> target,
                     std::shared_ptr<std::complex<float>> source);

    // Member variables used for compute
    unsigned int m_Np; //!< Last number of points computed
    float m_rmax;      //!< Maximum r at which to determine neighbors
    unsigned int m_l;  //!< Spherical harmonic l value.
    float m_rmin;      //!< Minimum r at which to determine neighbors (default 0)

    // Flags
    bool m_average; //!< Whether to take a second shell average (default false)
    bool m_norm;    //!< Whether to take the norm of the order parameter (default false)
    bool m_Wl;      //!< Whether to use the modified order parameter Wl (default false)

    std::shared_ptr<std::complex<float>> m_Qlmi; //!< Qlm for each particle i
    std::shared_ptr<std::complex<float>> m_Qlm;  //!< Normalized Qlm(Ave) for the whole system
    tbb::enumerable_thread_specific<std::complex<float>*> m_Qlm_local; //!< Thread-specific m_Qlm(Ave)
    std::shared_ptr<float> m_Qli;              //!< Ql locally invariant order parameter for each particle i
    std::shared_ptr<float> m_QliAve;           //!< Averaged Ql with 2nd neighbor shell for each particle i
    std::shared_ptr<complex<float>> m_QlmiAve; //!< Averaged Qlm with 2nd neighbor shell for each particle i
    std::shared_ptr<std::complex<float>> m_QlmAve;   //!< Normalized QlmiAve for the whole system
    float m_Norm;                                    //!< System normalized norm over all Qlm(Ave)
    std::complex<float> m_NormWl;                    //!< System normalized norm over all Wlm(Ave)
    std::shared_ptr<std::complex<float>> m_WliOrder; //!< Wl order parameter for each particle i
};

}; };  // end namespace freud::order
#endif // STEINHARDT_H
