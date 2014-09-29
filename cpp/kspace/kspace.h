#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "num_util.h"
#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

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

        // void set_K(float3* K, unsigned int NK)
        void set_K(vec3<float>* K, unsigned int NK)
            {
            m_K = K;
            m_NK = NK;
            }
        /*! Python wrapper to set_K
        \param K NK x 3 ndarray of K values to evaluate
        */
        void set_K_Py(boost::python::numeric::array K)
            {
            // validate input type and rank
            num_util::check_type(K, PyArray_FLOAT);
            num_util::check_rank(K, 2);
            // validate width of the 2nd dimension
            num_util::check_dim(K, 1, 3);
            unsigned int NK = num_util::shape(K)[0];
            // get the raw data pointers
            // float3* K_raw = (float3*) num_util::data(K);
            vec3<float>* K_raw = (vec3<float>*) num_util::data(K);
            set_K(K_raw, NK);
            }
        /*! Set particle positions and orientations
        \param position Np x 3 Array of particle position vectors
        \param orientation Np x 4 Array of particle orientation quaternions
        \param Np Number of particles
        */
        // void set_rq(unsigned int Np, float3* position, float4* orientation)
        void set_rq(unsigned int Np, vec3<float>* position, quat<float>* orientation)
            {
            m_Np = Np;
            m_r = position;
            m_q = orientation;
            }
        /*! Python wrapper to set_rq
        \param position Np x 3 ndrray of particle position vectors
        \param orientation Np x 4 ndrray of particle orientation quaternions
        */
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
            // float3* r_raw = (float3*) num_util::data(position);
            vec3<float>* r_raw = (vec3<float>*) num_util::data(position);
            // float4* q_raw = (float4*) num_util::data(orientation);
            quat<float>* q_raw = (quat<float>*) num_util::data(orientation);
            set_rq(Np, r_raw, q_raw);
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

    protected:
        boost::shared_array<float> m_S_Re;  //!< Real component of structure factor
        boost::shared_array<float> m_S_Im;  //!< Imaginary component of structure factor
        unsigned int m_NK;                  //!< number of K points evaluated
        unsigned int m_Np;                  //!< number of particles (length of r and q arrays)
        // float3* m_K;                        //!< array of K points
        // float3* m_r;                        //!< array of particle positions
        vec3<float>* m_K;                        //!< array of K points
        vec3<float>* m_r;                        //!< array of particle positions
        // float4* m_q;                        //!< array of particle orientations
        quat<float>* m_q;                        //!< array of particle orientations
        float m_density_Re;                 //!< real component of the scattering density
        float m_density_Im;                 //!< imaginary component of the scattering density
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
        float m_radius;                     //!< particle radius
        float m_volume;                     //!< particle volume
    };

//! Data structure for polyhedron vertices
/*! \ingroup hpmc_data_structs */
struct poly3d_param_t
    {
    std::vector< vec3<float> > vert;                    //!< Polyhedron vertices
    std::vector< std::vector<unsigned int> > facet;     //!< list of facets, which are lists of vertex indices
    std::vector< vec3<float> > norm;                    //!< normal unit vectors corresponding to facets
    std::vector< float > d;                             //!< distances of origin to facets
    std::vector< float > area;                          //!< pre-computed facet areas
    float volume;                                       //!< pre-computed polyhedron volume
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

        //! Set particle data structure
        // \param params data structure of necessary polyhedron information
        void set_params(const param_type& params);

    private:
        param_type m_params;        //!< polyhedron data structure
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_kspace();

}; }; // end namespace freud::kspace

#endif // _KSPACE_H__
