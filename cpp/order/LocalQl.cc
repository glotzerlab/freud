// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include "LocalQl.h"

using namespace std;

/*! \file LocalQl.cc
    \brief Compute a Ql per particle
*/

namespace freud { namespace order {

LocalQl::LocalQl(const box::Box& box, float rmax, unsigned int l, float rmin) : Steinhardt(box, rmax, l, rmin) {}

// Calculating Ylm using fsph module
void LocalQl::Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y)
    {
    if(Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    fsph::PointSPHEvaluator<float> sph_eval(m_l);

    unsigned int j(0);
    // old definition in compute (theta: 0...pi, phi: 0...2pi)
    // in fsph, the definition is flipped
    sph_eval.compute(theta, phi);

    for(typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(m_l, 0, true));
        iter != sph_eval.end(); ++iter)
        {
        Y[j] = *iter;
        ++j;
        }
    }

void LocalQl::compute(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    this->computeQl(nlist, points, Np);
    }

void LocalQl::computeAve(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    this->computeQlAve(nlist, points, Np);
    }

void LocalQl::computeNorm(const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;
    float normalizationfactor = 4*M_PI/(2*m_l+1);

    m_QliNorm = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    memset((void*)m_QliNorm.get(), 0, sizeof(float)*m_Np);

    // Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm.get()[k]/= m_Np;
        }

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            // Square by multiplying self w/ complex conj, then take real comp
            m_QliNorm.get()[i]+= abs( m_Qlm.get()[k] *
                                      conj(m_Qlm.get()[k]) );
            }
            m_QliNorm.get()[i]*=normalizationfactor;
            m_QliNorm.get()[i]=sqrt(m_QliNorm.get()[i]);
        }
    }

void LocalQl::computeAveNorm(const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;
    float normalizationfactor = 4*M_PI/(2*m_l+1);

    m_QliAveNorm = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    memset((void*)m_QliAveNorm.get(), 0, sizeof(float)*m_Np);

    //Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_AveQlm.get()[k]/= m_Np;
        }

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            //Square by multiplying self w/ complex conj, then take real comp
            m_QliAveNorm.get()[i]+= abs( m_AveQlm.get()[k] *
                                         conj(m_AveQlm.get()[k]) );
            }
            m_QliAveNorm.get()[i]*=normalizationfactor;
            m_QliAveNorm.get()[i]=sqrt(m_QliAveNorm.get()[i]);
        }
    }

}; }; // end namespace freud::order
