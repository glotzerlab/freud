#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"
#include "TensorMath.h"
#include "saruprng.h"

#include "NearestNeighbors.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _CUBATIC_ORDER_PARAMETER_H__
#define _CUBATIC_ORDER_PARAMETER_H__

/*! \file CubaticOrderParameter.h
    \brief Compute the cubatic order parameter for each particle
*/

namespace freud { namespace order {
//! Compute the cubatic order parameter for a set of points
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
        tensor4<float> m_gen_r4_tensor;
        unsigned int m_n;                //!< Last number of points computed
        unsigned int m_n_replicates;                //!< Last number of points computed

        float m_cubatic_order_parameter;
        quat<float> m_cubatic_orientation;
        std::shared_ptr<float> m_particle_order_parameter;
        tensor4<float> m_global_tensor;
        tensor4<float> m_cubatic_tensor;
        std::shared_ptr<float> m_particle_tensor;

        // saru rng
        Saru m_saru;
        unsigned int m_seed;
    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMETER_H__
