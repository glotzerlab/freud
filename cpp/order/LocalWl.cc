// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>

#include "LocalWl.h"

using namespace std;

/*! \file LocalWl.cc
    \brief Compute a Wl per particle.  Returns NaN if no neighbors.
*/

namespace freud { namespace order {

LocalWl::LocalWl(const box::Box& box, float rmax, unsigned int l, float rmin) :
    LocalQl(box, rmax, l, rmin)
    {
    m_normalizeWl = false;
    }

// Calculating Ylm using fsph module
void LocalWl::computeYlm(const float theta, const float phi, std::vector<std::complex<float> > &Ylm)
    {
    if (Ylm.size() != 2*m_l+1)
        {
        Ylm.resize(2*m_l+1);
        }

    fsph::PointSPHEvaluator<float> sph_eval(m_l);

    unsigned int j(0);
    // old definition in compute (theta: 0...pi, phi: 0...2pi)
    // in fsph, the definition is flipped
    sph_eval.compute(theta, phi);

    for (typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(m_l, 0, false));
         iter != sph_eval.end(); ++iter)
        {
        Ylm[(j+m_l) % (2*m_l+1)] = *iter;
        ++j;
        }
    for (unsigned int i = 1; i <= m_l; i++)
        {
        Ylm[-i+m_l] = Ylm[i+m_l];
        }
    }

void LocalWl::compute(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    if (m_Np != Np)
        {
        // Don't actually change the size, let the parent do that.
        m_Wli = std::shared_ptr<complex<float> >(new complex<float>[Np], std::default_delete<complex<float>[]>());
        }
    memset((void*) m_Wli.get(), 0, sizeof(complex<float>)*Np);

    // Call parent to compute Ql values used for calculating Wl.
    LocalQl::compute(nlist, points, Np);

    // This normalization happens in the Ql calculation but
    // not for Wl, so we need to undo it. In that calculation
    // the quantity is multiplied by the normalization factor
    // and then the result is square rooted, so here we just
    // divide by the square root.
    float normalizationfactor = sqrt(4*M_PI/(2*m_l+1));

    // Get wigner3j coefficients from wigner3j.cc
    m_wigner3jvalues = getWigner3j(m_l);

    for (unsigned int i = 0; i < m_Np; i++)
        {
        // Revert Ql normalization
        m_Qli.get()[i] /= normalizationfactor;

        // Wli calculation
        unsigned int counter = 0;
        for (unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for (unsigned int u2 = max(0, int(m_l)-int(u1)); u2 < min(3*m_l+1-u1, 2*m_l+1); ++u2)
                {
                const unsigned int index = (2*m_l+1)*i;
                const unsigned int u3 = 3*m_l-u1-u2;
                m_Wli.get()[i] += m_wigner3jvalues[counter] *
                                  m_Qlmi.get()[index + u1] *
                                  m_Qlmi.get()[index + u2] *
                                  m_Qlmi.get()[index + u3];
                counter++;
                }
            } // Ends loop for Wli calcs
        if (m_normalizeWl)
            {
            // Normalize
            m_Wli.get()[i] /= (m_Qli.get()[i] * m_Qli.get()[i] * m_Qli.get()[i]);
            }
        m_counter = counter;
        } // Ends loop over particles
    }

void LocalWl::computeAve(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    LocalQl::computeAve(nlist, points, Np);

    // Get wigner3j coefficients from wigner3j.cc
    m_wigner3jvalues = getWigner3j(m_l);

    m_AveWli = std::shared_ptr<complex<float> >(new complex<float>[m_Np], std::default_delete<complex<float>[]>());
    memset((void*) m_AveWli.get(), 0, sizeof(float)*m_Np);

    for (unsigned int i = 0; i < m_Np; i++)
        {
        // Ave Wli calculation
        unsigned int counter = 0;
        for (unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for (unsigned int u2 = max(0, int(m_l)-int(u1)); u2 < min(3*m_l+1-u1,2*m_l+1); ++u2)
                {
                const unsigned int index = (2*m_l+1)*i;
                const unsigned int u3 = 3*m_l-u1-u2;
                m_AveWli.get()[i] += m_wigner3jvalues[counter] *
                                     m_AveQlmi.get()[index + u1] *
                                     m_AveQlmi.get()[index + u2] *
                                     m_AveQlmi.get()[index + u3];
                counter++;
                }
            } // Ends loop for Norm Wli calcs
        m_counter = counter;
        } // Ends loop over particles
    }

void LocalWl::computeNorm(const vec3<float> *points, unsigned int Np)
    {
    // Get wigner3j coefficients from wigner3j.cc
    m_wigner3jvalues = getWigner3j(m_l);

    // Set local data size
    m_Np = Np;

    m_WliNorm = std::shared_ptr<complex<float> >(new complex<float>[m_Np], std::default_delete<complex<float>[]>());
    memset((void*) m_WliNorm.get(), 0, sizeof(complex<float>)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm.get()[k] /= m_Np;
        }

    for (unsigned int i = 0; i < m_Np; ++i)
        {
        // Norm Wli calculation
        unsigned int counter = 0;
        for (unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for (unsigned int u2 = max(0, int(m_l)-int(u1)); u2 < min(3*m_l+1-u1,2*m_l+1); ++u2)
                {
                unsigned int u3 = 3*m_l-u1-u2;
                m_WliNorm.get()[i] += m_wigner3jvalues[counter] *
                                      m_Qlm.get()[u1] *
                                      m_Qlm.get()[u2] *
                                      m_Qlm.get()[u3];
                counter++;
                }
            } // Ends loop for Norm Wli calcs
        m_counter = counter;
        } // Ends loop over particles
    }

void LocalWl::computeAveNorm(const vec3<float> *points, unsigned int Np)
    {

    // Get wigner3j coefficients from wigner3j.cc
    m_wigner3jvalues = getWigner3j(m_l);

    // Set local data size
    m_Np = Np;

    m_WliAveNorm = std::shared_ptr<complex<float> >(new complex<float>[m_Np], std::default_delete<complex<float>[]>());
    memset((void*) m_WliAveNorm.get(), 0, sizeof(complex<float>)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_AveQlm.get()[k] /= m_Np;
        }

    for (unsigned int i = 0; i < m_Np; ++i)
        {
        // AveNorm Wli calculation
        unsigned int counter = 0;
        for (unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for (unsigned int u2 = max(0, int(m_l)-int(u1)); u2 < min(3*m_l+1-u1,2*m_l+1); ++u2)
                {
                unsigned int u3 = 3*m_l-u1-u2;
                m_WliAveNorm.get()[i] += m_wigner3jvalues[counter] *
                                         m_AveQlm.get()[u1] *
                                         m_AveQlm.get()[u2] *
                                         m_AveQlm.get()[u3];
                counter++;
                }
            } // Ends loop for Norm Wli calcs
        m_counter = counter;
        } // Ends loop over particles
    }

}; }; // end namespace freud::order
