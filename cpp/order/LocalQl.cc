// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "LocalQl.h"

#include <stdexcept>
#include <complex>
//#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;

/*! \file LocalQl.cc
    \brief Compute a Ql per particle
*/

namespace freud { namespace order {

LocalQl::LocalQl(const box::Box& box, float rmax, unsigned int l, float rmin)
    :m_box(box), m_rmax(rmax), m_l(l), m_rmin(rmin)
    {
    if (m_rmax < 0.0f or m_rmin < 0.0f)
        throw invalid_argument("rmin and rmax must be positive!");
    if (m_rmin >= m_rmax)
        throw invalid_argument("rmin should be smaller than rmax!");
    if (m_l < 2)
        throw invalid_argument("l must be two or greater!");
    }

/* void LocalQl::Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y)
    {
    if(Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    for(int m = -m_l; m <=0; m++)
        //Doc for boost spherical harmonic
        //http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html
        // theta = colatitude = -1..Pi
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


// void LocalQl::compute(const float3 *points, unsigned int Np)
void LocalQl::compute(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    float rminsq = m_rmin * m_rmin;
    float rmaxsq = m_rmax * m_rmax;
    float normalizationfactor = 4*M_PI/(2*m_l+1);

    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
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
        //get cell point is in
        // float3 ref = points[i];
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
                // float dx = float(points[j].x - ref.x);
                // float dy = float(points[j].y - ref.y);
                // float dz = float(points[j].z - ref.z);

                // float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                vec3<float> delta = m_box.wrap(points[j] - ref);
                // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
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
                    LocalQl::Ylm(theta, phi,Y);  //Fill up Ylm vector

                    for(unsigned int k = 0; k < (2*m_l+1); ++k)
                        {
                        m_Qlmi.get()[(2*m_l+1)*i+k]+=Y[k];
                        }
                    neighborcount++;
                    }
            } //End loop going over neighbor cells (and thus all neighboring particles);
            //Normalize!
            for(unsigned int k = 0; k < (2*m_l+1); ++k)
                {
                m_Qlmi.get()[(2*m_l+1)*i+k]/= neighborcount;
                m_Qli.get()[i]+= abs( m_Qlmi.get()[(2*m_l+1)*i+k]*conj(m_Qlmi.get()[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
                m_Qlm.get()[k]+= m_Qlmi.get()[(2*m_l+1)*i+k];
                }
        m_Qli.get()[i]*=normalizationfactor;
        m_Qli.get()[i]=sqrt(m_Qli.get()[i]);
        } //Ends loop over particles i for Qlmi calcs
    }

// void LocalQl::computeAve(const float3 *points, unsigned int Np)
void LocalQl::computeAve(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    //Set local data size
    m_Np = Np;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    float rminsq = m_rmin * m_rmin;
    float rmaxsq = m_rmax * m_rmax;
    float normalizationfactor = 4*M_PI/(2*m_l+1);

    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_AveQlmi = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*m_Np], std::default_delete<complex<float>[]>());
    m_AveQli = std::shared_ptr<float>(new float[m_Np], std::default_delete<float[]>());
    m_AveQlm = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)], std::default_delete<complex<float>[]>());
    memset((void*)m_AveQlmi.get(), 0, sizeof(complex<float>)*(2*m_l+1)*m_Np);
    memset((void*)m_AveQli.get(), 0, sizeof(float)*m_Np);
    memset((void*)m_AveQlm.get(), 0, sizeof(complex<float>)*(2*m_l+1));

    size_t bond(0);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        // float3 ref = points[i];
        vec3<float> ref = points[i];
        unsigned int neighborcount=1;

        for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
                const unsigned int n1(neighbor_list[2*bond + 1]);
                {
                vec3<float> ref1 = points[n1];
                if (n1 == i)
                    {
                        continue;
                    }
                // rij = rj - ri, from i pointing to j.
                // float dx = float(points[n1].x - ref.x);
                // float dy = float(points[n1].y - ref.y);
                // float dz = float(points[n1].z - ref.z);

                // float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                vec3<float> delta = m_box.wrap(ref1 - ref);
                // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                float rsq = dot(delta, delta);

                if (rsq < rmaxsq and rsq > rminsq)
                    {
                    size_t neighborhood_bond(nlist->find_first_index(n1));
                    for(; neighborhood_bond < nlist->getNumBonds() && neighbor_list[2*neighborhood_bond] == n1; ++neighborhood_bond)
                    {
                    const unsigned int j(neighbor_list[2*neighborhood_bond + 1]);
                            {
                            if (n1 == j)
                                {
                                    continue;
                                }
                            // rij = rj - ri, from i pointing to j.
                            // float dx1 = float(points[j].x - ref1.x);
                            // float dy1 = float(points[j].y - ref1.y);
                            // float dz1 = float(points[j].z - ref1.z);

                            // float3 delta1 = m_box.wrap(make_float3(dx1, dy1, dz1));
                            vec3<float> delta1 = m_box.wrap(points[j] - ref1);
                            // float rsq1 = delta1.x*delta1.x + delta1.y*delta1.y + delta1.z*delta1.z;
                            float rsq1 = dot(delta1, delta1);

                            if (rsq1 < rmaxsq and rsq1 > rminsq)
                                {
                                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                                    {
                                    //adding all the Qlm of the neighbors
                                    // change to Index?
                                    // Seg fault is here
                                    // m_Qlmi is not instantiated in this loop method, compute must be called first?
                                    m_AveQlmi.get()[(2*m_l+1)*i+k] += m_Qlmi.get()[(2*m_l+1)*j+k];
                                    }
                                neighborcount++;
                                }
                            }
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
void LocalQl::computeNorm(const vec3<float> *points, unsigned int Np)
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
            m_QliAveNorm.get()[i]+= abs( m_AveQlm.get()[k]*conj(m_AveQlm.get()[k]) ); //Square by multiplying self w/ complex conj, then take real comp
            }
            m_QliAveNorm.get()[i]*=normalizationfactor;
            m_QliAveNorm.get()[i]=sqrt(m_QliAveNorm.get()[i]);
        }
    }


// void LocalQl::computePy(boost::python::numeric::array points)
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

// void LocalQl::computeAvePy(boost::python::numeric::array points)
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

// void LocalQl::computeNormPy(boost::python::numeric::array points)
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

// void LocalQl::computeAveNormPy(boost::python::numeric::array points)
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

// void export_LocalQl()
//     {
//     class_<LocalQl>("LocalQl", init<box::Box&, float, unsigned int, optional<float> >())
//         .def("getBox", &LocalQl::getBox, return_internal_reference<>())
//         .def("setBox", &LocalQl::setBox)
//         .def("compute", &LocalQl::computePy)
//         .def("computeAve", &LocalQl::computeAvePy)
//         .def("computeNorm", &LocalQl::computeNormPy)
//         .def("computeAveNorm", &LocalQl::computeAveNormPy)
//         .def("getQl", &LocalQl::getQlPy)
//         .def("getAveQl", &LocalQl::getAveQlPy)
//         .def("getQlNorm", &LocalQl::getQlNormPy)
//         .def("getQlAveNorm", &LocalQl::getQlAveNormPy)
//         ;
//     }

}; }; // end namespace freud::localqi
