// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include "Steinhardt.h"

using namespace std;

/*! \file Steinhardt.cc
    \brief Compute some variant of Steinhardt order parameter.
*/

namespace freud {
namespace order {

Steinhardt::~Steinhardt() {}

void Steinhardt::computeQl(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    //Set local data size
    m_Np = Np;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    float rminsq = m_rmin * m_rmin;
    float rmaxsq = m_rmax * m_rmax;
    float normalizationfactor = 4*M_PI/(2*m_l+1);

    // newmanrs: For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_Qlmi = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*m_Np], std::default_delete<complex<float>[]>());
    m_Qli = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    m_Qlm = std::shared_ptr<complex<float> >(new complex<float>[2*m_l+1], std::default_delete<complex<float>[]>());
    memset((void*)m_Qlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*)m_Qli.get(), 0, sizeof(float)*m_Np);
    memset((void*)m_Qlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));

    size_t bond(0);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        // Get cell point is in
        vec3<float> ref = points[i];
        unsigned int neighborcount=0;

        for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const unsigned int j(neighbor_list[2*bond + 1]);

            if (i == j)
                {
                continue;
                }
            // rij = rj - ri, from i pointing to j.
            vec3<float> delta = m_box.wrap(points[j] - ref);
            float rsq = dot(delta, delta);

            if (rsq < rmaxsq and rsq > rminsq)
                {
                // phi is usually in range 0..2Pi, but
                // it only appears in Ylm as exp(im\phi),
                // so range -Pi..Pi will give same results.
                float phi = atan2(delta.y,delta.x);      //-Pi..Pi
                float theta = acos(delta.z / sqrt(rsq)); //0..Pi
                // if the points are directly on top of each other for whatever reason,
                // theta should be zero instead of nan.

                if (rsq == float(0))
                    {
                    theta = 0;
                    }

                std::vector<std::complex<float> > Y;
                this->Ylm(theta, phi,Y);  //Fill up Ylm vector

                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                    {
                    m_Qlmi.get()[(2*m_l+1)*i+k]+=Y[k];
                    }
                neighborcount++;
                }
            } // End loop going over neighbor cells (and thus all neighboring particles);
            //Normalize!
            for(unsigned int k = 0; k < (2*m_l+1); ++k)
                {
                m_Qlmi.get()[(2*m_l+1)*i+k]/= neighborcount;
                //Square by multiplying self w/ complex conj, then take real comp
                m_Qli.get()[i]+= abs( m_Qlmi.get()[(2*m_l+1)*i+k] *
                                      conj(m_Qlmi.get()[(2*m_l+1)*i+k]) );
                m_Qlm.get()[k]+= m_Qlmi.get()[(2*m_l+1)*i+k];
                }
        m_Qli.get()[i]*=normalizationfactor;
        m_Qli.get()[i]=sqrt(m_Qli.get()[i]);
        } // Ends loop over particles i for Qlmi calcs
    }
}; // end namespace freud::order
}; // end namespace freud
