// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include "LocalWl.h"
#include "wigner3j.h"
#include <stdexcept>
#include <complex>
#include <algorithm>
#include <cstring>

using namespace std;

/*! \file LocalWl.cc
    \brief Compute a Wl per particle.  Returns NaN if no neighbors.
*/

namespace freud { namespace order {

LocalWl::LocalWl(const box::Box& box, float rmax, unsigned int l)
    :m_box(box), m_rmax(rmax), m_l(l)
    {
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive!");
    if (m_l < 2)
        throw invalid_argument("l must be two or greater (and even)!");
    if (m_l%2 == 1)
        {
        fprintf(stderr,"Current value of m_l is %d\n",m_l);
        throw invalid_argument("This method requires even values of l!");
        }
    m_normalizeWl = false;
    }

// Calculating Ylm using fsph module
void LocalWl::Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y)
    {
    if (Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    fsph::PointSPHEvaluator<float> sph_eval(m_l);

    unsigned int j(0);
    // old definition in compute (theta: 0...pi, phi: 0...2pi)
    // in fsph, the definition is flipped
    sph_eval.compute(theta, phi);

    for (typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(m_l, 0, false));
        iter != sph_eval.end(); ++iter)
        {
        Y[(j+m_l) % (2*m_l+1)] = *iter;
        ++j;
        }
    for (unsigned int i = 1; i <=m_l; i++)
        Y[-i+m_l] = Y[i+m_l];
    }

void LocalWl::compute(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    // Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<float> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    // Set local data size
    m_Np = Np;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    float rmaxsq = m_rmax * m_rmax;

    // newmanrs: For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_Qlmi = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*m_Np], std::default_delete<complex<float>[]>());
    m_Qli = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    m_Wli = std::shared_ptr<complex<float> >(new complex<float>[m_Np], std::default_delete<complex<float>[]>());
    m_Qlm = std::shared_ptr<complex<float> >(new complex<float>[2*m_l+1], std::default_delete<complex<float>[]>());
    memset((void*)m_Qlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*)m_Wli.get(), 0, sizeof(complex<float>)*m_Np);
    memset((void*)m_Qlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));
    memset((void*)m_Qli.get(), 0, sizeof(float)*m_Np);

    size_t bond(0);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        // Get cell point is in
        vec3<float> ref = points[i];
        unsigned int neighborcount=0;

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const unsigned int j(neighbor_list[2*bond + 1]);
                {
                if (i == j)
                    {
                    continue;
                    }
                // rij = rj - ri, from i pointing to j.
                vec3<float> delta = m_box.wrap(points[j] - ref);
                float rsq = dot(delta, delta);

                if (rsq < rmaxsq)
                    {
                    float phi = atan2(delta.y,delta.x);      //0..2Pi
                    float theta = acos(delta.z / sqrt(rsq)); //0..Pi

                    std::vector<std::complex<float> > Y;
                    LocalWl::Ylm(theta, phi,Y); // Fill up Ylm vector
                    for(unsigned int k = 0; k < (2*m_l+1); ++k)
                        {
                        // change to Index later
                        m_Qlmi.get()[(2*m_l+1)*i+k]+=Y[k];
                        }
                    neighborcount++;
                    }
                }
            } // End loop going over neighbor cells (and thus all neighboring particles);
            // Normalize!
            for(unsigned int k = 0; k < (2*m_l+1); ++k)
                {
                m_Qlmi.get()[(2*m_l+1)*i+k]/= neighborcount;
                m_Qli.get()[i]+=abs( m_Qlmi.get()[(2*m_l+1)*i+k]*conj(m_Qlmi.get()[(2*m_l+1)*i+k]) );
                m_Qlm.get()[k]+= m_Qlmi.get()[(2*m_l+1)*i+k];
                } //Ends loop over particles i for Qlmi calcs
        m_Qli.get()[i]=sqrt(m_Qli.get()[i]);//*sqrt(m_Qli[i])*sqrt(m_Qli[i]);//Normalize factor for Wli

        // Wli calculation
        unsigned int counter = 0;
        for (unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
                {
                unsigned int u3 = 3*m_l-u1-u2;
                m_Wli.get()[i] += m_wigner3jvalues[counter]*m_Qlmi.get()[(2*m_l+1)*i+u1]*m_Qlmi.get()[(2*m_l+1)*i+u2]*m_Qlmi.get()[(2*m_l+1)*i+u3];
                counter+=1;
                }
            } // Ends loop for Wli calcs
        if (m_normalizeWl)
            {
            m_Wli.get()[i]/=(m_Qli.get()[i]*m_Qli.get()[i]*m_Qli.get()[i]);//Normalize
            }
        m_counter = counter;
        }
    }

void LocalWl::computeAve(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {

    // Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<float> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    // Set local data size
    m_Np = Np;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    float rmaxsq = m_rmax * m_rmax;

    // Maybe consider if Np != m_Np, we could not reallocate these
    m_AveQlmi = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*m_Np], std::default_delete<complex<float>[]>());
    m_AveQlm = std::shared_ptr<complex<float> > (new complex<float> [(2*m_l+1)], std::default_delete<complex<float>[]>());
    m_AveWli = std::shared_ptr<complex<float> >(new complex<float> [m_Np], std::default_delete<complex<float>[]>());
    memset((void*)m_AveQlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*)m_AveQlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));
    memset((void*)m_AveWli.get(), 0, sizeof(float)*m_Np);

    size_t bond(0);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        // Get cell point is in
        vec3<float> ref = points[i];
        unsigned int neighborcount=1;

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const unsigned int n1(neighbor_list[2*bond + 1]);
                {
                vec3<float> ref1 = points[n1];
                if (n1 == i)
                    {
                    continue;
                    }
                // rij = rj - ri, from i pointing to j.
                vec3<float> delta = m_box.wrap(points[n1] - ref);
                float rsq = dot(delta, delta);
                if (rsq < rmaxsq)
                    {
                    size_t neighborhood_bond(nlist->find_first_index(n1));
                    for (; neighborhood_bond < nlist->getNumBonds() && neighbor_list[2*neighborhood_bond] == n1; ++neighborhood_bond)
                        {
                        const unsigned int j(neighbor_list[2*neighborhood_bond + 1]);
                            {
                            if (n1 == j)
                                {
                                continue;
                                }
                            // rij = rj - ri, from i pointing to j.
                            vec3<float> delta1 = m_box.wrap(points[j] - ref1);
                            float rsq1 = dot(delta1, delta1);

                            if (rsq1 < rmaxsq)
                                {
                                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                                    {
                                    m_AveQlmi.get()[(2*m_l+1)*i+k] += m_Qlmi.get()[(2*m_l+1)*j+k];
                                    }
                                neighborcount++;
                                }
                            }
                        }
                    }
                }
            }
         // Normalize!
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            m_AveQlmi.get()[(2*m_l+1)*i+k] += m_Qlmi.get()[(2*m_l+1)*i+k];
            m_AveQlmi.get()[(2*m_l+1)*i+k]/= neighborcount;
            m_AveQlm.get()[k] += m_AveQlmi.get()[(2*m_l+1)*i+k];
            }
        // Ave Wli calculation
        unsigned int counter = 0;
        for(unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
                {
                unsigned int u3 = 3*m_l-u1-u2;
                m_AveWli.get()[i]+= m_wigner3jvalues[counter]*m_AveQlmi.get()[(2*m_l+1)*i+u1]*m_AveQlmi.get()[(2*m_l+1)*i+u2]*m_AveQlmi.get()[(2*m_l+1)*i+u3];
                counter+=1;
                }
            } // Ends loop for Norm Wli calcs
        m_counter = counter;

        } // Ends loop over particles i for Qlmi calcs
    }

void LocalWl::computeNorm(const vec3<float> *points, unsigned int Np)
    {

    // Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<float> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    // Set local data size
    m_Np = Np;

    m_WliNorm = std::shared_ptr<complex<float> >(new complex<float>[m_Np], std::default_delete<complex<float>[]>());
    memset((void*)m_WliNorm.get(), 0, sizeof(complex<float>)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm.get()[k]/= m_Np;
        }

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        // Norm Wli calculation
        unsigned int counter = 0;
        for(unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
                {
                unsigned int u3 = 3*m_l-u1-u2;
                m_WliNorm.get()[i] += m_wigner3jvalues[counter] *
                                      m_Qlm.get()[u1] *
                                      m_Qlm.get()[u2] *
                                      m_Qlm.get()[u3];
                counter+=1;
                }
            } // Ends loop for Norm Wli calcs
        m_counter = counter;
        }
    }

void LocalWl::computeAveNorm(const vec3<float> *points, unsigned int Np)
    {

    // Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<float> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    // Set local data size
    m_Np = Np;

    m_WliAveNorm = std::shared_ptr<complex<float> >(new complex<float>[m_Np], std::default_delete<complex<float>[]>());
    memset((void*)m_WliAveNorm.get(), 0, sizeof(complex<float>)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_AveQlm.get()[k]/= m_Np;
        }

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        // AveNorm Wli calculation
        unsigned int counter = 0;
        for(unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
            {
            for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
                {
                unsigned int u3 = 3*m_l-u1-u2;
                m_WliAveNorm.get()[i] += m_wigner3jvalues[counter] *
                                         m_AveQlm.get()[u1] *
                                         m_AveQlm.get()[u2] *
                                         m_AveQlm.get()[u3];
                counter+=1;
                }
            } // Ends loop for Norm Wli calcs
        m_counter = counter;
        }
    }

}; }; // end namespace freud::order
