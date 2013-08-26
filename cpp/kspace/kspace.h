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
        boost::shared_array<float> m_S_Re;    //!< Real component of structure factor
        boost::shared_array<float> m_S_Im;    //!< Imaginary component of structure factor
        unsigned int m_NK;                    //!< number of K points evaluated
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_kspace();

}; }; // end namespace freud::kspace

#endif // _KSPACE_H__
