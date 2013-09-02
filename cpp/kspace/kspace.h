#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "num_util.h"
#include "HOOMDMath.h"

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

        void set_K(float3* K, unsigned int NK)
            {
            m_K = K;
            m_NK = NK;
            }
        void set_K_Py(boost::python::numeric::array K)
            {
            // validate input type and rank
            num_util::check_type(K, PyArray_FLOAT);
            num_util::check_rank(K, 2);
            // validate width of the 2nd dimension
            num_util::check_dim(K, 1, 3);
            unsigned int NK = num_util::shape(K)[0];
            // get the raw data pointers
            float3* K_raw = (float3*) num_util::data(K);
            set_K(K_raw, NK);
            }
        void set_rq(unsigned int Np, float3* position, float4* orientation)
            {
            m_Np = Np;
            m_r = position;
            m_q = orientation;
            }
        void set_rq_Py(boost::python::numeric::array position,
                    boost::python::numeric::array orientation)
            {
            // validate input type and rank
            num_util::check_type(position, PyArray_FLOAT);
            num_util::check_rank(position, 2);
            num_util::check_type(orientation, PyArray_FLOAT);
            num_util::check_rank(orientation, 2);

            // validate width of the 2nd dimension
            num_util::check_dim(position, 1, 3);
            unsigned int Np = num_util::shape(position)[0];

            num_util::check_dim(orientation, 1, 4);
            // Make sure orientation is same length as position
            num_util::check_dim(orientation, 0, Np);

            // get the raw data pointers
            float3* r_raw = (float3*) num_util::data(position);
            float4* q_raw = (float4*) num_util::data(orientation);
            set_rq(Np, r_raw, q_raw);
            }
        void set_scale(float scale)
            {
            m_scale = scale;
            }
        void set_density(std::complex<float> density)
            {
            m_density_Re = density.real();
            m_density_Im = density.imag();
            }

        //! Perform transform and store result internally
        /*
        \param K Array of K values to evaluate
        \param NK Number of K values
        \param r Np x 3 Array of particle position vectors
        \param q Np x 4 Array of particle orientation quaternions
        \param Np Number of particles
        \param scale Scaling factor to apply to r
        \param density_Re Real component of the scattering density
        \param density_Im Imaginary component of the scattering density
        */
        virtual void compute(const float3 *K,
                     const unsigned int NK,
                     const float3 *r,
                     const float4 *q,
                     const unsigned int Np,
                     const float scale,
                     const float density_Re,
                     const float density_Im
                     );

        //! Python wrapper for compute method
        /*! Provide a consistent interface for the Python module.
            \param K Nx3 ndarray of K points
            \param r Nx3 ndarray of particle positions
            \param orientation Nx4 ndarray of orientation quaternions (unused)
            \param scale scaling factor to apply to r
            \param density complex valued scattering density
        */
        virtual void computePy(boost::python::numeric::array K,
                       boost::python::numeric::array r,
                       boost::python::numeric::array orientation,
                       float scale,
                       std::complex<float> density
                       );

        //! C++ interface to return the FT values
        boost::shared_array< std::complex<float> > getFT()
            {
            boost::shared_array< std::complex<float> > arr;
            arr = boost::shared_array< std::complex<float> >(new std::complex<float>[m_NK]);
            for(unsigned int i = 0; i < m_NK; i++)
                {
                arr[i] = std::complex<float>(m_S_Re[i], m_S_Im[i]);
                }
            return arr;
            }

        //! Python interface to return the FT values (returns a copy)
        boost::python::numeric::array getFTPy()
            {
            // FT must be created as a placeholder so that the boost::shared_array returned by getFT().get()
            // does not go out of scope and get garbage collected before num_util::makeNum is called.
            boost::shared_array< std::complex<float> > FT;
            FT = getFT();
            std::complex<float> *arr = FT.get();
            return num_util::makeNum(arr, m_NK);
            }

    private:
        boost::shared_array<float> m_S_Re;  //!< Real component of structure factor
        boost::shared_array<float> m_S_Im;  //!< Imaginary component of structure factor
        unsigned int m_NK;                  //!< number of K points evaluated
        unsigned int m_Np;                  //!< number of particles (length of r and q arrays)
        float3* m_K;                    //!< array of K points
        float3* m_r;                    //!< array of particle positions
        float4* m_q;                    //!< array of particle orientations
        float m_scale;                      //!< length scale (to be multiplied by spatial dimensions)
        float m_density_Re;                 //!< real component of the scattering density
        float m_density_Im;                 //!< imaginary component of the scattering density
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_kspace();

}; }; // end namespace freud::kspace

#endif // _KSPACE_H__
