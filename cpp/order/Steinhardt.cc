// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "Steinhardt.h"

using namespace std;
using namespace tbb;

/*! \file Steinhardt.cc
    \brief Computes variants of Steinhardt order parameters.
*/

namespace freud { namespace order {

// Calculating Ylm using fsph module
void Steinhardt::computeYlm(const float theta, const float phi, std::vector<std::complex<float> > &Ylm)
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

    if (m_Wl)
        {
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
    else
        {
        for (typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(m_l, 0, true));
            iter != sph_eval.end(); ++iter)
            {
            Ylm[j] = *iter;
            ++j;
            }
        }
    }

template <typename T>
std::shared_ptr<T> Steinhardt::makeArray(size_t size)
    {
    return std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
    }

void Steinhardt::reallocateArrays(unsigned int Np)
    {
    m_Qlmi = Steinhardt::makeArray<complex<float> >((2*m_l+1) * Np);
    m_Qli = Steinhardt::makeArray<float>(Np);
    m_Qlm = Steinhardt::makeArray<complex<float> >(2*m_l+1);

    if (m_average)
        {
        m_QlmiAve = Steinhardt::makeArray<complex<float> >((2*m_l+1) * Np);
        m_QliAve = Steinhardt::makeArray<float>(Np);
        m_QlmAve = Steinhardt::makeArray<complex<float> >(2*m_l+1);
        }

    if (m_Wl)
        {
        if (m_average && m_norm)
            {
            m_WliAveNorm = Steinhardt::makeArray<complex<float> >(Np);
            }
        else if (m_average)
            {
            m_WliAve = Steinhardt::makeArray<complex<float> >(Np);
            }
        else if (m_norm)
            {
            m_WliNorm = Steinhardt::makeArray<complex<float> >(Np);
            }
        else
            {
            m_Wli = Steinhardt::makeArray<complex<float> >(Np);
            }
        }
    else
        {
        if (m_average && m_norm)
            {
            m_QliAveNorm = Steinhardt::makeArray<float>(Np);
            }
        else if (m_norm)
            {
            m_QliNorm = Steinhardt::makeArray<float>(Np);
            }
        }
    }

void Steinhardt::compute(const box::Box& box, const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    // Conditionally reinitialize arrays if size differs from previous call.
    if (m_Np != Np)
        {
        m_Np = Np;
        Steinhardt::reallocateArrays(Np);
        }

    // Computes the base Q required for each specialized order parameter
    Steinhardt::baseCompute(box, nlist, points);

    if (m_average)
        {
        Steinhardt::computeAve(box, nlist, points);
        }

    if (m_Wl)
        {
        if (m_average && m_norm)
            {
            Steinhardt::computeAveNormWl();
            }
        else if (m_norm)
            {
            Steinhardt::computeNormWl();
            }
        else if (m_average)
            {
            Steinhardt::computeAveWl();
            }
        else
            {
            Steinhardt::computeWl();
            }
        }
    else
        {
        if (m_average && m_norm)
            {
            Steinhardt::computeAveNorm();
            }
        else if (m_norm)
            {
            Steinhardt::computeNorm();
            }
        }
    }

void Steinhardt::baseCompute(const box::Box& box, const locality::NeighborList *nlist, const vec3<float> *points)
    {
    nlist->validate(m_Np, m_Np);

    memset((void*) m_Qlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*) m_Qli.get(), 0, sizeof(float)*m_Np);

    parallel_for(tbb::blocked_range<size_t>(0, m_Np),
        [=] (const blocked_range<size_t>& r)
        {
        const float rminsq = m_rmin * m_rmin;
        const float rmaxsq = m_rmax * m_rmax;
        const float normalizationfactor = 4*M_PI/(2*m_l+1);
        const size_t *neighbor_list(nlist->getNeighbors());

        bool Qlm_exists;
        m_Qlm_local.local(Qlm_exists);
        if (!Qlm_exists)
            {
            m_Qlm_local.local() = new complex<float> [2*m_l+1];
            memset((void*) m_Qlm_local.local(), 0, sizeof(complex<float>) * (2*m_l+1));
            }

        size_t bond(nlist->find_first_index(r.begin()));
        // for each reference point
        for (size_t i = r.begin(); i != r.end(); i++)
            {
            unsigned int neighborcount(0);
            const vec3<float> ref(points[i]);
            for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                {
                const unsigned int j(neighbor_list[2*bond + 1]);

                if (i == j)
                    {
                    continue;
                    }

                const vec3<float> delta = box.wrap(points[j] - ref);
                const float rsq = dot(delta, delta);

                if (rsq < rmaxsq && rsq > rminsq)
                    {
                    // phi is usually in range 0..2Pi, but
                    // it only appears in Ylm as exp(im\phi),
                    // so range -Pi..Pi will give same results.
                    float phi = atan2(delta.y, delta.x);     // -Pi..Pi
                    float theta = acos(delta.z / sqrt(rsq)); // 0..Pi

                    // If the points are directly on top of each other,
                    // theta should be zero instead of nan.
                    if (rsq == float(0))
                        {
                        theta = 0;
                        }

                    std::vector<std::complex<float> > Ylm(2*m_l+1);
                    this->computeYlm(theta, phi, Ylm);  // Fill up Ylm

                    for (unsigned int k = 0; k < Ylm.size(); ++k)
                        {
                        m_Qlmi.get()[(2*m_l+1)*i+k] += Ylm[k];
                        }
                    neighborcount++;
                    }
                } // End loop going over neighbor bonds

                // Normalize!
                for (unsigned int k = 0; k < (2*m_l+1); ++k)
                    {
                    const unsigned int index = (2*m_l+1) * i + k;
                    m_Qlmi.get()[index] /= neighborcount;
                    // Add the norm, which is the (complex) squared magnitude
                    m_Qli.get()[i] += norm(m_Qlmi.get()[index]);
                    m_Qlm_local.local()[k] += m_Qlmi.get()[index];
                    }
            m_Qli.get()[i] *= normalizationfactor;
            m_Qli.get()[i] = sqrt(m_Qli.get()[i]);
            } // Ends loop over particles i for Qlmi calcs
        });
    reduce();
    }

void Steinhardt::computeAve(const box::Box& box, const locality::NeighborList *nlist, const vec3<float> *points)
    {
    const size_t *neighbor_list(nlist->getNeighbors());

    memset((void*) m_QlmiAve.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*) m_QliAve.get(), 0, sizeof(float)*m_Np);
    memset((void*) m_QlmAve.get(), 0, sizeof(complex<float>)*(2*m_l+1));

    const float rminsq = m_rmin * m_rmin;
    const float rmaxsq = m_rmax * m_rmax;
    const float normalizationfactor = 4*M_PI/(2*m_l+1);

    size_t bond(0);

    // Can be parallelized
    for (unsigned int i = 0; i < m_Np; i++)
        {
        const vec3<float> ri = points[i];
        unsigned int neighborcount(1);

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const unsigned int n(neighbor_list[2*bond + 1]);
            const vec3<float> rn = points[n];

            if (n == i)
                {
                continue;
                }

            // rin = rn - ri, from i pointing to j.
            const vec3<float> rin = box.wrap(rn - ri);
            const float rinsq = dot(rin, rin);

            if (rinsq < rmaxsq && rinsq > rminsq)
                {
                size_t neighborhood_bond(nlist->find_first_index(n));
                for (; neighborhood_bond < nlist->getNumBonds() && neighbor_list[2*neighborhood_bond] == n; ++neighborhood_bond)
                    {
                    const unsigned int j(neighbor_list[2*neighborhood_bond + 1]);

                    if (n == j)
                        {
                        continue;
                        }

                    // rnj = rj - rn, from n pointing to j.
                    const vec3<float> rnj = box.wrap(points[j] - rn);
                    const float rnjsq = dot(rnj, rnj);

                    if (rnjsq < rmaxsq && rnjsq > rminsq)
                        {
                        for (unsigned int k = 0; k < (2*m_l+1); ++k)
                            {
                            // Adding all the Qlm of the neighbors
                            m_QlmiAve.get()[(2*m_l+1)*i+k] += m_Qlmi.get()[(2*m_l+1)*j+k];
                            }
                        neighborcount++;
                        }
                    } // End loop over particle neighbor's bonds
                }
            } // End loop over particle's bonds

        // Normalize!
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            const unsigned int index = (2*m_l+1) * i + k;
            // Adding the Qlm of the particle i itself
            m_QlmiAve.get()[index] += m_Qlmi.get()[index];
            m_QlmiAve.get()[index] /= neighborcount;
            m_QlmAve.get()[k] += m_QlmiAve.get()[index];
            // Add the norm, which is the complex squared magnitude
            m_QliAve.get()[i] += norm(m_Qlmi.get()[index]);
            }
        m_QliAve.get()[i] *= normalizationfactor;
        m_QliAve.get()[i] = sqrt(m_QliAve.get()[i]);
        } // Ends loop over particles i for Qlmi calcs
    }

void Steinhardt::computeNorm()
    {
    memset((void*) m_QliNorm.get(), 0, sizeof(float)*m_Np);

    const float normalizationfactor = 4*M_PI/(2*m_l+1);

    // Average Q_lm over all particles, which was calculated in compute
    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm.get()[k] /= m_Np;
        }

    for (unsigned int i = 0; i < m_Np; ++i)
        {
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            // Add the norm, which is the complex squared magnitude
            m_QliNorm.get()[i] += norm(m_Qlm.get()[k]);
            }
        m_QliNorm.get()[i] *= normalizationfactor;
        m_QliNorm.get()[i] = sqrt(m_QliNorm.get()[i]);
        }
    }

void Steinhardt::computeAveNorm()
    {
    memset((void*) m_QliAveNorm.get(), 0, sizeof(float)*m_Np);

    const float normalizationfactor = 4*M_PI/(2*m_l+1);

    // Average Q_lm over all particles, which was calculated in compute
    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_QlmAve.get()[k] /= m_Np;
        }

    for (unsigned int i = 0; i < m_Np; ++i)
        {
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            // Add the norm, which is the complex squared magnitude
            m_QliAveNorm.get()[i] += norm(m_QlmAve.get()[k]);
            }
        m_QliAveNorm.get()[i] *= normalizationfactor;
        m_QliAveNorm.get()[i] = sqrt(m_QliAveNorm.get()[i]);
        }
    }

void Steinhardt::aggregateWl(std::shared_ptr<complex<float> > target,
                             std::shared_ptr<complex<float> > source,
                             bool per_particle)
    {
    // Get wigner3j coefficients
    m_wigner3jvalues = getWigner3j(m_l);

    for (unsigned int i = 0; i < m_Np; i++)
        {
        unsigned int counter = 0;
        for (unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for (unsigned int u2 = max(0, int(m_l)-int(u1)); u2 < min(3*m_l+1-u1, 2*m_l+1); ++u2)
                {
                const unsigned int particle_index = per_particle ? (2*m_l+1)*i : 0;
                const unsigned int u3 = 3*m_l-u1-u2;
                target.get()[i] += m_wigner3jvalues[counter] *
                    source.get()[particle_index + u1] *
                    source.get()[particle_index + u2] *
                    source.get()[particle_index + u3];
                counter++;
                }
            } // Ends loop over Wigner 3j coefficients
        } // Ends loop over particles
   }

void Steinhardt::computeWl()
    {
    memset((void*) m_Wli.get(), 0, sizeof(complex<float>)*m_Np);

    Steinhardt::aggregateWl(m_Wli, m_Qlmi, true);
    }

void Steinhardt::computeAveWl()
    {
    memset((void*) m_WliAve.get(), 0, sizeof(float)*m_Np);

    Steinhardt::aggregateWl(m_WliAve, m_QlmiAve, true);
    }

void Steinhardt::computeNormWl()
    {
    memset((void*) m_WliNorm.get(), 0, sizeof(complex<float>)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm.get()[k] /= m_Np;
        }

    Steinhardt::aggregateWl(m_WliNorm, m_Qlm, false);
    }

void Steinhardt::computeAveNormWl()
    {
    memset((void*) m_WliAveNorm.get(), 0, sizeof(complex<float>)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_QlmAve.get()[k] /= m_Np;
        }

    Steinhardt::aggregateWl(m_WliAveNorm, m_QlmAve, false);
    }

void Steinhardt::reduce()
    {
    memset((void*) m_Qlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));
    parallel_for(tbb::blocked_range<size_t>(0, 2*m_l+1),
        [=] (const blocked_range<size_t>& r)
        {
        for (size_t i = r.begin(); i != r.end(); i++)
            {
            for (tbb::enumerable_thread_specific<complex<float> *>::const_iterator Qlm_local = m_Qlm_local.begin();
                 Qlm_local != m_Qlm_local.end(); Qlm_local++)
                {
                m_Qlm.get()[i] += (*Qlm_local)[i];
                }
            }
        });
    }

}; }; // end namespace freud::order
