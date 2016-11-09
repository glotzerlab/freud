// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "LocalQlNear.h"

#include <stdexcept>
#include <complex>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;

/*! \file LocalQl.cc
    \brief Compute a Ql per particle using N nearest neighbors instead of r_cut
*/

namespace freud { namespace order {

LocalQlNear::LocalQlNear(const box::Box& box, float rmax, unsigned int l, unsigned int kn)
    :m_box(box), m_rmax(rmax), m_l(l), m_k(kn)
    {
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive!");
    if (m_l < 2)
        throw invalid_argument("l must be two or greater!");
    // if (m_l%2 == 1)
    //     {
    //     fprintf(stderr,"Current value of m_l is %d\n",m_l);
    //     throw invalid_argument("This method requires even values of l!");
    //     }
    m_nn = new locality::NearestNeighbors(m_rmax, m_k);
    }

LocalQlNear::~LocalQlNear()
    {
    delete m_nn;
    }

/*
void LocalQlNear::Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y)
    {
    if(Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    for(int m = -m_l; m <=0; m++)
        //Doc for boost spherical harmonic
        //http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html
        // theta = colatitude = 0..Pi
        // Phi = azimuthal (longitudinal) 0..2pi).
        Y[m+m_l]= boost::math::spherical_harmonic(m_l, m, theta, phi);

    // This states that Y(l,+m) = Y(l,-m).
    // Actually, Y(l,m) = (-1)^m * complex.conjugate[Y(l,-m)]
    // This doesn't matter when you take the norm, however.
    for(unsigned int i = 1; i <= m_l; i++)
        Y[i+m_l] = Y[-i+m_l];
    }
*/

// Calculating Ylm using fsph module
void LocalQlNear::Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y)
    {
    if (Y.size() != 2*m_l+1)
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

// void LocalQl::compute(const float3 *points, unsigned int Np)
void LocalQlNear::compute(const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;

    //Initialize neighbor list
    m_nn->compute(m_box,points,Np,points,Np);

    float normalizationfactor = 4*M_PI/(2*m_l+1);


    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_Qlmi = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*m_Np], std::default_delete<complex<float>[]>());
    m_Qli = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    m_Qlm = std::shared_ptr<complex<float> >(new complex<float>[2*m_l+1], std::default_delete<complex<float>[]>());
    memset((void*)m_Qlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*)m_Qli.get(), 0, sizeof(float)*m_Np);
    memset((void*)m_Qlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        // float3 ref = points[i];
        vec3<float> ref = points[i];
        std::shared_ptr<unsigned int> neighbors = m_nn->getNeighbors(i);

        //loop over neighboring cells
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            unsigned int j = neighbors.get()[neigh_idx];

            //compute r between the two particles.
            vec3<float> delta = m_box.wrap(points[j] - ref);
            float rsq = dot(delta, delta);

            if (rsq > 1e-6)
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
                LocalQlNear::Ylm(theta, phi,Y);  //Fill up Ylm vector
                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                    {
                    m_Qlmi.get()[(2*m_l+1)*i+k]+=Y[k];
                    }
                }
            } //End loop going over neighbor cells (and thus all neighboring particles);
            //Normalize!
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            m_Qlmi.get()[(2*m_l+1)*i+k]/= m_k;
            m_Qli.get()[i]+= abs( m_Qlmi.get()[(2*m_l+1)*i+k]*conj(m_Qlmi.get()[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
            m_Qlm.get()[k]+= m_Qlmi.get()[(2*m_l+1)*i+k];
            }
        m_Qli.get()[i]*=normalizationfactor;
        m_Qli.get()[i]=sqrt(m_Qli.get()[i]);
        } //Ends loop over particles i for Qlmi calcs
    }

// void LocalQl::computeAve(const float3 *points, unsigned int Np)
void LocalQlNear::computeAve(const vec3<float> *points, unsigned int Np)
    {
    //Set local data size
    m_Np = Np;

    // compute the neighbor list
    m_nn->compute(m_box,points,Np,points,Np);

    float normalizationfactor = 4*M_PI/(2*m_l+1);


    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_AveQlmi = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*m_Np], std::default_delete<complex<float>[]>());
    m_AveQli = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    m_AveQlm = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)], std::default_delete<complex<float>[]>());
    memset((void*)m_AveQlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*)m_AveQli.get(), 0, sizeof(float)*m_Np);
    memset((void*)m_AveQlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        vec3<float> ref = points[i];
        unsigned int neighborcount = 1;
        std::shared_ptr<unsigned int> neighbors = m_nn->getNeighbors(i);
        //loop over neighbors
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            //get cell points of 1st neighbor
            unsigned int j = neighbors.get()[neigh_idx];

            if (j == i)
                {
                continue;
                }

            vec3<float> ref1 = points[j];
            vec3<float> delta = m_box.wrap(points[j] - ref);

            float rsq = dot(delta, delta);
            if (rsq > 1e-6)
                {
                std::shared_ptr<unsigned int> neighbors_2 = m_nn->getNeighbors(j);

                for (unsigned int neigh1_idx = 0; neigh1_idx < m_k; neigh1_idx++)
                    {
                    //get cell points of 2nd neighbor
                    unsigned int n1 = neighbors_2.get()[neigh1_idx];
                    if (n1 == j)
                        {
                        continue;
                        }

                    vec3<float> delta1 = m_box.wrap(points[n1] - ref1);
                    float rsq1 = dot(delta1, delta1);

                    if (rsq1 > 1e-6)
                        {
                        for(unsigned int k = 0; k < (2*m_l+1); ++k)
                            {
                            //adding all the Qlm of the neighbors
                            // change to Index?
                            m_AveQlmi.get()[(2*m_l+1)*i+k] += m_Qlmi.get()[(2*m_l+1)*n1+k];
                            }
                        neighborcount++;
                        }
                    }
                } //End loop going over neighbor cells (and thus all neighboring particles);
            }
        //Normalize!
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
                //adding the Qlm of the particle i itself
                m_AveQlmi.get()[(2*m_l+1)*i+k] += m_Qlmi.get()[(2*m_l+1)*i+k];
                m_AveQlmi.get()[(2*m_l+1)*i+k]/= neighborcount;
                m_AveQlm.get()[k]+= m_AveQlmi.get()[(2*m_l+1)*i+k];
                m_AveQli.get()[i]+= abs( m_AveQlmi.get()[(2*m_l+1)*i+k]*conj(m_AveQlmi.get()[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
            }
        m_AveQli.get()[i]*=normalizationfactor;
        m_AveQli.get()[i]=sqrt(m_AveQli.get()[i]);
        } //Ends loop over particles i for Qlmi calcs
    }

// void LocalQl::computeNorm(const float3 *points, unsigned int Np)
void LocalQlNear::computeNorm(const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;
    float normalizationfactor = 4*M_PI/(2*m_l+1);

    m_QliNorm = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    memset((void*)m_QliNorm.get(), 0, sizeof(float)*m_Np);

    //Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm.get()[k]/= m_Np;
        }

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            m_QliNorm.get()[i]+= abs( m_Qlm.get()[k]*conj(m_Qlm.get()[k]) ); //Square by multiplying self w/ complex conj, then take real comp
            }
            m_QliNorm.get()[i]*=normalizationfactor;
            m_QliNorm.get()[i]=sqrt(m_QliNorm.get()[i]);
        }
    }

// void LocalQl::computeAveNorm(const float3 *points, unsigned int Np)
void LocalQlNear::computeAveNorm(const vec3<float> *points, unsigned int Np)
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
            m_QliAveNorm.get()[i]+= abs( m_AveQlm.get()[k]*conj(m_AveQlm.get()[k]) ); //Square by multiplying self w/ complex conj, then take real comp
             }
            m_QliAveNorm.get()[i]*=normalizationfactor;
            m_QliAveNorm.get()[i]=sqrt(m_QliAveNorm.get()[i]);
        }
    }


// void LocalQlNear::computePy(boost::python::numeric::array points)
//     {
//     //validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers and compute the cell list
//     // float3* points_raw = (float3*) num_util::data(points);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     compute(points_raw, Np);
//     }

// void LocalQlNear::computeAvePy(boost::python::numeric::array points)
//     {
//     //validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers and compute the cell list
//     // float3* points_raw = (float3*) num_util::data(points);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     compute(points_raw, Np);
//     computeAve(points_raw, Np);
//     }

// void LocalQlNear::computeNormPy(boost::python::numeric::array points)
//     {
//     //validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers and compute the cell list
//     // float3* points_raw = (float3*) num_util::data(points);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     compute(points_raw, Np);
//     computeNorm(points_raw, Np);
//     }

// void LocalQlNear::computeAveNormPy(boost::python::numeric::array points)
//     {
//     //validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers and compute the cell list
//     // float3* points_raw = (float3*) num_util::data(points);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     compute(points_raw, Np);
//     computeAve(points_raw, Np);
//     computeAveNorm(points_raw, Np);
//     }

// void export_LocalQlNear()
//     {
//     class_<LocalQlNear>("LocalQlNear", init<box::Box&, float, unsigned int, optional<unsigned int> >())
//         .def("getBox", &LocalQlNear::getBox, return_internal_reference<>())
//         .def("setBox", &LocalQlNear::setBox)
//         .def("compute", &LocalQlNear::computePy)
//         .def("computeAve", &LocalQlNear::computeAvePy)
//         .def("computeNorm", &LocalQlNear::computeNormPy)
//         .def("computeAveNorm", &LocalQlNear::computeAveNormPy)
//         .def("getQl", &LocalQlNear::getQlPy)
//         .def("getAveQl", &LocalQlNear::getAveQlPy)
//         .def("getQlNorm", &LocalQlNear::getQlNormPy)
//         .def("getQlAveNorm", &LocalQlNear::getQlAveNormPy)
//         ;
//     }

}; }; // end namespace freud::order
