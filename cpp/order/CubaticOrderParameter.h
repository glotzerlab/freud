// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CUBATIC_ORDER_PARAMETER_H
#define CUBATIC_ORDER_PARAMETER_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "VectorMath.h"
#include "saruprng.h"

/*! \file CubaticOrderParameter.h
    \brief Compute the cubatic order parameter for each particle.
*/

namespace freud { namespace order {

//! Helper 4th-order tensor class for cubatic calculations.
/*! The cubatic order parameter involves many calculations that use a 4th-order
 *  tensor of size 3 in each dimension. This simple helper class functions as a
 *  data container that also defines various operators to simplify the code.
 */
struct tensor4
{
    tensor4();
    tensor4(vec3<float> _vector);
    tensor4 operator+=(const tensor4& b);
    tensor4 operator+=(const float& b);
    tensor4 operator-(const tensor4& b);
    tensor4 operator-=(const tensor4& b);
    tensor4 operator*(const float& b);
    tensor4 operator*=(const float& b);

    void reset();

    float data[81];
};

float dot(const tensor4& a, const tensor4& b);








//! Compute the cubatic order parameter for a set of points
/*!
 */
class CubaticOrderParameter
{
public:
    //! Constructor
    CubaticOrderParameter(float t_initial, float t_final, float scale, float* r4_tensor,
                          unsigned int replicates, unsigned int seed);

    //! Destructor
    ~CubaticOrderParameter() {}

    //! Reset the bond order array to all zeros
    void reset();

    //! Compute the cubatic order parameter
    void compute(quat<float>* orientations, unsigned int n);

    //! Calculate the cubatic tensor
    void calcCubaticTensor(float* cubatic_tensor, quat<float> orientation);

    void calcCubaticOrderParameter(float& cubatic_order_parameter, float* cubatic_tensor);

    //! Get a reference to the last computed cubatic order parameter
    float getCubaticOrderParameter()
    {
        return m_cubatic_order_parameter;
    }

    quat<float> calcRandomQuaternion(Saru& saru, float angle_multiplier);

    std::shared_ptr<float> getParticleCubaticOrderParameter()
    {
        return m_particle_order_parameter;
    }

    std::shared_ptr<float> getParticleTensor()
    {
        return m_particle_tensor;
    }

    std::shared_ptr<float> getGlobalTensor()
    {
        memcpy(m_sp_global_tensor.get(), (void*) &m_global_tensor.data, sizeof(float) * 81);
        return m_sp_global_tensor;
    }

    std::shared_ptr<float> getCubaticTensor()
    {
        memcpy(m_sp_cubatic_tensor.get(), (void*) &m_cubatic_tensor.data, sizeof(float) * 81);
        return m_sp_cubatic_tensor;
    }

    std::shared_ptr<float> getGenR4Tensor()
    {
        memcpy(m_sp_gen_r4_tensor.get(), (void*) &m_gen_r4_tensor.data, sizeof(float) * 81);
        return m_sp_gen_r4_tensor;
    }

    unsigned int getNumParticles()
    {
        return m_n;
    }

    float getTInitial()
    {
        return m_t_initial;
    }

    float getTFinal()
    {
        return m_t_final;
    }

    float getScale()
    {
        return m_scale;
    }

    quat<float> getCubaticOrientation()
    {
        return m_cubatic_orientation;
    }

private:
    float m_t_initial;         //!< Initial temperature for simulated annealing.
    float m_t_final;           //!< Final temperature for simulated annealing.
    float m_scale;             //!< Scaling factor to reduce temperature.
    unsigned int m_n;          //!< Last number of points computed.
    unsigned int m_replicates; //!< Number of replicates.

    float m_cubatic_order_parameter;   //!< The value of the order parameter.
    quat<float> m_cubatic_orientation; //!< The cubatic orientation.

    tensor4 m_gen_r4_tensor;
    tensor4 m_global_tensor;
    tensor4 m_cubatic_tensor;

    std::shared_ptr<float> m_particle_order_parameter; //!< The per-particle value of the order parameter.
    std::shared_ptr<float>
        m_sp_global_tensor; //!< Shared pointer for global tensor, only used to return values to Python.
    std::shared_ptr<float>
        m_sp_cubatic_tensor; //!< Shared pointer for cubatic tensor, only used to return values to Python.
    std::shared_ptr<float> m_particle_tensor;
    std::shared_ptr<float>
        m_sp_gen_r4_tensor; //!< Shared pointer for r4 tensor, only used to return values to Python.

    unsigned int m_seed; //!< Random seed
};

}; }; // end namespace freud::order

#endif // CUBATIC_ORDER_PARAMETER_H
