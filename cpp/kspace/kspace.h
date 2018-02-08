// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <complex>
#include <vector>

#include "HOOMDMath.h"
#include "VectorMath.h"
#ifndef _KSPACE_H__
#define _KSPACE_H__

//! \package freud::kspace Provides tools for calculating density in reciprocal space
/*! For analyzing periodicity or simulating diffraction patterns
*/
namespace freud { namespace kspace {
/*! Compute the Fourier transform of a set of delta peaks at a list of K points.
Also serves as the base class for Fourier transform calculators of non-delta form factors
*/
class FTdelta
    {
    public:
        //! Constructor
        FTdelta();

        //! Destructor
        virtual ~FTdelta();

        void set_K(vec3<float>* K, unsigned int NK)
            {
            m_NK = NK;
            m_K.resize(NK);
            std::copy(K, K+NK, m_K.begin());

            // initialize output array
            m_arr = std::shared_ptr< std::complex<float> >(new std::complex<float>[m_NK], std::default_delete<std::complex<float>[]>());
            }

        /*! Set particle positions and orientations
        \param position Np x 3 Array of particle position vectors
        \param orientation Np x 4 Array of particle orientation quaternions
        \param Np Number of particles
        */
        void set_rq(unsigned int Np, vec3<float>* position, quat<float>* orientation)
            {
            m_Np = Np;
            m_r.resize(Np);
            m_q.resize(Np);
            std::copy(position, position + Np, m_r.begin());
            std::copy(orientation, orientation + Np, m_q.begin());
            }

        /*! Set scattering density
        \param density complex value of scattering density
        */
        void set_density(const std::complex<float> density)
            {
            m_density_Re = density.real();
            m_density_Im = density.imag();
            }

        //! Perform transform and store result internally
        virtual void compute();

        //! Python wrapper for compute method
        virtual void computePy();

        //! C++ interface to return the FT values
        std::shared_ptr< std::complex<float> > getFT()
            {
            for(unsigned int i = 0; i < m_NK; i++)
                {
                m_arr.get()[i] = std::complex<float>(m_S_Re.get()[i], m_S_Im.get()[i]);
                }
            return m_arr;
            }

    protected:
        std::shared_ptr< std::complex<float> > m_arr;
        std::shared_ptr<float> m_S_Re;  //!< Real component of structure factor
        std::shared_ptr<float> m_S_Im;  //!< Imaginary component of structure factor
        unsigned int m_NK;              //!< number of K points evaluated
        unsigned int m_Np;              //!< number of particles (length of r and q arrays)
        std::vector<vec3<float> > m_K;  //!< array of K points
        std::vector<vec3<float> > m_r;  //!< array of particle positions
        std::vector<quat<float> > m_q;  //!< array of particle orientations
        float m_density_Re;             //!< real component of the scattering density
        float m_density_Im;             //!< imaginary component of the scattering density
    };

class FTsphere: public FTdelta
    {
    public:
        //! Constructor
        FTsphere();

        //! Perform transform and store result internally
        virtual void compute();

        //! Set particle volume according to radius
        void set_radius(const float radius)
            {
            m_radius = radius;
            m_volume = 4.0f * radius*radius*radius / 3.0f;
            }

    private:
        float m_radius;  //!< particle radius
        float m_volume;  //!< particle volume
    };

//! Data structure for polyhedron vertices
/*! \ingroup hpmc_data_structs */
struct poly3d_param_t
    {
    std::vector< vec3<float> > vert;                 //!< Polyhedron vertices
    std::vector< std::vector<unsigned int> > facet;  //!< list of facets, which are lists of vertex indices
    std::vector< vec3<float> > norm;                 //!< normal unit vectors corresponding to facets
    std::vector< float > area;                       //!< pre-computed facet areas
    std::vector< float > d;                          //!< distances of origin to facets
    float volume;                                    //!< pre-computed polyhedron volume
    };

class FTpolyhedron: public FTdelta
    {
    public:
        typedef poly3d_param_t param_type;

        //! Constructor
        FTpolyhedron();

        //! Perform transform and store result internally
        //! Note that for a scale factor, lambda, affecting the size of the scatterer,
        //! S_lambda(k) == lambda**3 * S(lambda * k)
        virtual void compute();

        void set_params(unsigned int nvert,
                       vec3<float>* vert,
                       unsigned int nfacet,
                       unsigned int *facet_offs,
                       unsigned int *facet,
                       vec3<float>* norm,
                       float *d,
                       float * area,
                       float volume);

    private:
        param_type m_params;  //!< polyhedron data structure
    };

}; }; // end namespace freud::kspace

#endif // _KSPACE_H__
