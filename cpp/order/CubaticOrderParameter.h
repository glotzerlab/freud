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

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class CubaticOrderParameter
    {
    public:
        //! Constructor
        CubaticOrderParameter();

        CubaticOrderParameter(float t_initial, float t_final, float scale, float* r4_tensor);

        //! Destructor
        ~CubaticOrderParameter();

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

        std::shared_ptr<float> getParticleCubaticOrderParameter();

        std::shared_ptr<float> getParticleTensor();

        std::shared_ptr<float> getGlobalTensor();

        std::shared_ptr<float> getCubaticTensor();

        std::shared_ptr<float> getGenR4Tensor();

        unsigned int getNumParticles();


        // std::shared_ptr<float> getParticleCubaticOrderParameter();

    private:

        float m_t_initial;
        float m_t_final;
        float m_scale;
        std::shared_ptr<float> m_gen_r4_tensor;
        unsigned int m_n;                //!< Last number of points computed
        unsigned int m_n_replicates;                //!< Last number of points computed

        float m_cubatic_order_parameter;
        quat<float> m_cubatic_orientation;
        std::shared_ptr<float> m_particle_order_parameter;
        std::shared_ptr<float> m_global_tensor;
        std::shared_ptr<float> m_cubatic_tensor;
        std::shared_ptr<float> m_particle_tensor;

        // serial rng
        std::random_device m_rd;
        std::mt19937 m_gen;
        std::uniform_real_distribution<float> m_theta_dist;
        std::uniform_real_distribution<float> m_phi_dist;
        std::uniform_real_distribution<float> m_angle_dist;
        // boost::shared_array<float> m_particle_order_parameter;         //!< phi order array computed
        // tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
        // tbb::enumerable_thread_specific<std::random_device *> m_local_rd;
        // tbb::enumerable_thread_specific<std::mt19937 *> m_local_gen;
        // tbb::enumerable_thread_specific<std::uniform_real_distribution *> m_local_dist;

    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMETER_H__
