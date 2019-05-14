// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CUBATIC_ORDER_PARAMETER_H
#define CUBATIC_ORDER_PARAMETER_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "TensorMath.h"
#include "saruprng.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

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

    private:

        float m_t_initial;                                   //!< Initial temperature for simulated annealing.
        float m_t_final;                                     //!< Final temperature for simulated annealing.
        float m_scale;                                       //!< Scaling factor to reduce temperature.
        unsigned int m_n;                                    //!< Last number of points computed.
        unsigned int m_replicates;                           //!< Number of replicates.

        float m_cubatic_order_parameter;                     //!< The value of the order parameter.
        quat<float> m_cubatic_orientation;                   //!< The cubatic orientation.

        tensor4<float> m_gen_r4_tensor;
        tensor4<float> m_global_tensor;
        tensor4<float> m_cubatic_tensor;

        std::shared_ptr<float> m_particle_order_parameter;   //!< The per-particle value of the order parameter.
        std::shared_ptr<float> m_sp_global_tensor;           //!< Shared pointer for global tensor, only used to return values to Python.
        std::shared_ptr<float> m_sp_cubatic_tensor;          //!< Shared pointer for cubatic tensor, only used to return values to Python.
        std::shared_ptr<float> m_particle_tensor;
        std::shared_ptr<float> m_sp_gen_r4_tensor;           //!< Shared pointer for r4 tensor, only used to return values to Python.

        // saru rng
        Saru m_saru;
        unsigned int m_seed;
    };

}; }; // end namespace freud::order

#endif // CUBATIC_ORDER_PARAMETER_H
