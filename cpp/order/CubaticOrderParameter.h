#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>
#include <random>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"
#include "saruprng.h"

#include "NearestNeighbors.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _CUBATIC_ORDER_PARAMETER_H__
#define _CUBATIC_ORDER_PARAMETER_H__

/*! \file CubaticOrderParameter.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

// helper functions...these will need cleaned up better
void tensorProduct(float *tensor, vec3<float> vector);
float tensorDot(float *tensor_a, float *tensor_b);
void tensorMult(float *tensor_out, float a);
void tensorDiv(float *tensor_out, float a);
void tensorSub(float *tensor_out, float *tensor_i, float *tensor_j);
void tensorAdd(float *tensor_out, float *tensor_i, float *tensor_j);

template < class Real >
struct tensor4
    {
    tensor4()
        {
        memset((void*)&data, 0, sizeof(float)*81);
        }
    tensor4(vec3<Real> &_vector)
        {
        unsigned int cnt = 0;
        float v[3];
        v[0] = _vector.x;
        v[1] = _vector.y;
        v[2] = _vector.z;
        for (unsigned int i = 0; i < 3; i++)
            {
            float v_i = v[i];
            for (unsigned int j = 0; j < 3; j++)
                {
                float v_j = v[j];
                for (unsigned int k = 0; k < 3; k++)
                    {
                    float v_k = v[k];
                    for (unsigned int l = 0; l < 3; l++)
                        {
                        float v_l = v[l];
                        data[cnt] = v_i * v_j * v_k * v_l;
                        cnt++;
                        }
                    }
                }
            }
        }
    tensor4(Real (&_data)[81])
        {
        memcpy((void*)data, (void*)_data, sizeof(float)*81);
        }
    tensor4(float* _data)
        {
        memcpy((void*)data, (void*)_data, sizeof(float)*81);
        }
    Real data[81];
    };

template < class Real >
tensor4<Real> operator+(const tensor4<Real>& a, const tensor4<Real>& b)
    {
    tensor4<Real> c;
    for (unsigned int i=0; i<81; i++)
        {
        c.data[i] = a.data[i] + b.data[i];
        }
    return c;
    }

template < class Real >
tensor4<Real> operator+(const tensor4<Real>& a, const Real& b)
    {
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] + b;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator+=(tensor4<Real>& a, const tensor4<Real>& b)
    {
    for (unsigned int i=0; i<81; i++)
        {
        a.data[i] += b.data[i];
        }
    return a;
    }

template < class Real >
tensor4<Real> operator+=(tensor4<Real>& a, const Real& b)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] += b;
        }
    return a;
    }

template < class Real >
tensor4<Real> operator-(const tensor4<Real>& a, const tensor4<Real>& b)
    {
    tensor4<Real> c;
    for (unsigned int i=0; i<81; i++)
        {
        c.data[i] = a.data[i] - b.data[i];
        }
    return c;
    }

template < class Real >
tensor4<Real> operator-(const tensor4<Real>& a, const Real& b)
    {
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] - b;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator-=(tensor4<Real>& a, const tensor4<Real>& b)
    {
    for (unsigned int i=0; i<81; i++)
        {
        a.data[i] -= b.data[i];
        }
    return a;
    }

template < class Real >
tensor4<Real> operator-=(tensor4<Real>& a, const Real& b)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] -= b;
        }
    }

template < class Real >
float dot(const tensor4<Real>& a, const tensor4<Real>& b)
    {
    Real c = 0;
    for (unsigned int i = 0; i < 81; i++)
        {
        c += a.data[i] * b.data[i];
        }
    return c;
    }

template < class Real >
tensor4<Real> operator*(const tensor4<Real>& a, const Real& b)
    {
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] * b;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator/(const tensor4<Real>& a, const Real& b)
    {
    Real b_inv = 1.0/b;
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] * b_inv;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator*=(tensor4<Real>& a, const Real& b)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] *= b;
        }
    }

template < class Real >
tensor4<Real> operator/=(tensor4<Real>& a, const Real& b)
    {
    Real b_inv = 1.0/b;
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] *= b_inv;
        }
    }

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class CubaticOrderParameter
    {
    public:
        //! Constructor
        CubaticOrderParameter(float t_initial, float t_final, float scale, float* r4_tensor, unsigned int n_replicates, unsigned int seed);

        //! Destructor
        // ~CubaticOrderParameter();

        //! Reset the bond order array to all zeros
        void resetCubaticOrderParameter(quat<float> orientation);

        //! accumulate the bond order
        void compute(quat<float> *orientations,
                     unsigned int n,
                     unsigned int n_replicates);

        // calculate the cubatic tensor
        void calcCubaticTensor(float *cubatic_tensor, quat<float> orientation);

        void calcCubaticOrderParameter(float &cubatic_order_parameter, float *cubatic_tensor);

        void reduceCubaticOrderParameter();

        //! Get a reference to the last computed rdf
        float getCubaticOrderParameter();

        quat<float> calcRandomQuaternion(Saru &saru, float angle_multiplier);

        std::shared_ptr<float> getParticleCubaticOrderParameter();

        std::shared_ptr<float> getParticleTensor();

        std::shared_ptr<float> getGlobalTensor();

        std::shared_ptr<float> getCubaticTensor();

        std::shared_ptr<float> getGenR4Tensor();

        unsigned int getNumParticles();

        float getTInitial();

        float getTFinal();

        float getScale();

        quat<float> getCubaticOrientation();


        // std::shared_ptr<float> getParticleCubaticOrderParameter();

    private:

        float m_t_initial;
        float m_t_final;
        float m_scale;
        // std::shared_ptr<float> m_gen_r4_tensor;
        tensor4<float> m_gen_r4_tensor;
        unsigned int m_n;                //!< Last number of points computed
        unsigned int m_n_replicates;                //!< Last number of points computed

        float m_cubatic_order_parameter;
        quat<float> m_cubatic_orientation;
        std::shared_ptr<float> m_particle_order_parameter;
        // std::shared_ptr<float> m_global_tensor;
        tensor4<float> m_global_tensor;
        // std::shared_ptr<float> m_cubatic_tensor;
        tensor4<float> m_cubatic_tensor;
        std::shared_ptr<float> m_particle_tensor;

        // serial rng
        std::random_device m_rd;
        std::mt19937 m_gen;
        std::uniform_real_distribution<float> m_theta_dist;
        std::uniform_real_distribution<float> m_phi_dist;
        std::uniform_real_distribution<float> m_angle_dist;
        Saru m_saru;
        unsigned int m_seed;
        // boost::shared_array<float> m_particle_order_parameter;         //!< phi order array computed
        // tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
        // tbb::enumerable_thread_specific<std::random_device *> m_local_rd;
        // tbb::enumerable_thread_specific<std::mt19937 *> m_local_gen;
        // tbb::enumerable_thread_specific<std::uniform_real_distribution *> m_local_dist;

    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMETER_H__
