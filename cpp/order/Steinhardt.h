// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STEINHARDT_H
#define STEINHARDT_H

#include <complex>
#include <memory>
#include <tbb/tbb.h>
#include <ManagedArray.h>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"
#include "ThreadStorage.h"
#include "Wigner3j.h"

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
     *  \param l Spherical harmonic number l.
     *           Must be a positive number.
     */
    Steinhardt(unsigned int l, bool average = false, bool Wl = false, bool weighted = false)
        : m_Np(0), m_l(l), m_num_ms(2*l+1), m_average(average), m_Wl(Wl), m_weighted(weighted), m_Qlm_local(2 * l + 1)
    {
        if (m_l < 2)
            throw std::invalid_argument("Steinhardt requires l must be two or greater.");
    }

    //! Empty destructor
    virtual ~Steinhardt() {};

    //! Get the number of particles used in the last compute
    unsigned int getNP() const
    {
        return m_Np;
    }

    //! Get the last calculated order parameter
    const util::ManagedArray<float> &getOrder() const
    {
        if (m_Wl)
        {
            return m_Wli;
        }
        else
        {
            return getQl();
        }
    }

    //! Get the last calculated Ql
    const util::ManagedArray<float> &getQl() const
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

    //! Get norm
    float getNorm() const
    {
        return m_norm;
    }

    //!< Whether to take a second shell average
    bool isAverage() const
    {
        return m_average;
    }

    //!< Whether to use the third-order invariant Wl
    bool isWl() const
    {
        return m_Wl;
    }

    //!< Whether to use neighbor weights in computing Qlmi
    bool isWeighted() const
    {
        return m_weighted;
    }

    //! Compute the order parameter
    virtual void compute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

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
    void baseCompute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

    //! Calculates the neighbor average Ql order parameter
    void computeAve(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

    //! Normalize the order parameter
    float normalize();

    //! Sum over Wigner 3j coefficients to compute third-order invariants
    //  Wl from second-order invariants Ql
    void aggregateWl(util::ManagedArray<float> &target,
                     util::ManagedArray<std::complex<float>> &source);

    // Member variables used for compute
    unsigned int m_Np; //!< Last number of points computed
    unsigned int m_l;  //!< Spherical harmonic l value.
    unsigned int m_num_ms; //!< The number of magnetic quantum numbers (2*m_l+1).

    // Flags
    bool m_average; //!< Whether to take a second shell average (default false)
    bool m_Wl;      //!< Whether to use the third-order invariant Wl (default false)
    bool m_weighted;      //!< Whether to use neighbor weights in computing Qlmi (default false)

    util::ManagedArray<std::complex<float>> m_Qlmi; //!< Qlm for each particle i
    util::ManagedArray<std::complex<float>> m_Qlm;  //!< Normalized Qlm(Ave) for the whole system
    util::ThreadStorage<std::complex<float>> m_Qlm_local; //!< Thread-specific m_Qlm(Ave)
    util::ManagedArray<float> m_Qli;              //!< Ql locally invariant order parameter for each particle i
    util::ManagedArray<float> m_QliAve;           //!< Averaged Ql with 2nd neighbor shell for each particle i
    util::ManagedArray<complex<float>> m_QlmiAve; //!< Averaged Qlm with 2nd neighbor shell for each particle i
    util::ManagedArray<std::complex<float>> m_QlmAve;   //!< Normalized QlmiAve for the whole system
    float m_norm;                                    //!< System normalized order parameter
    util::ManagedArray<float> m_Wli; //!< Wl order parameter for each particle i, also used for Wl averaged data
};

}; };  // end namespace freud::order
#endif // STEINHARDT_H
