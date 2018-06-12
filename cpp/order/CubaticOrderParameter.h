// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "box.h"
#include "VectorMath.h"
#include "TensorMath.h"
#include "saruprng.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

#ifndef _CUBATIC_ORDER_PARAMETER_H__
#define _CUBATIC_ORDER_PARAMETER_H__

/*! \file CubaticOrderParameter.h
    \brief Compute the cubatic order parameter for each particle.
*/

namespace freud { namespace order {
//! Compute the cubatic order parameter for a set of points
/*!
*/
class CubaticOrderParameter
    {
    public:
        //! Constructor
        CubaticOrderParameter(float t_initial, float t_final, float scale, float* r4_tensor, unsigned int replicates, unsigned int seed);

        //! Destructor
        ~CubaticOrderParameter();

        //! Reset the bond order array to all zeros
        void reset();

        //! Compute the cubatic order parameter
        void compute(quat<float> *orientations,
                     unsigned int n,
                     unsigned int replicates);

        //! Calculate the cubatic tensor
        void calcCubaticTensor(float *cubatic_tensor, quat<float> orientation);

        void calcCubaticOrderParameter(float &cubatic_order_parameter, float *cubatic_tensor);

        void reduceCubaticOrderParameter();

        //! Get a reference to the last computed cubatic order parameter
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
        unsigned int m_replicates;       //!< Number of replicates

        float m_cubatic_order_parameter;
        quat<float> m_cubatic_orientation;
        std::shared_ptr<float> m_particle_order_parameter;
        tensor4<float> m_global_tensor;
        std::shared_ptr<float> m_sp_global_tensor;
        tensor4<float> m_cubatic_tensor;
        std::shared_ptr<float> m_sp_cubatic_tensor;
        std::shared_ptr<float> m_particle_tensor;
        std::shared_ptr<float> m_sp_gen_r4_tensor;

        // saru rng
        Saru m_saru;
        unsigned int m_seed;
    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMETER_H__
