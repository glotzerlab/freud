// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CUBATIC_ORDER_PARAMETER_H
#define CUBATIC_ORDER_PARAMETER_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "VectorMath.h"
#include "saruprng.h"
#include "ManagedArray.h"

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
    float &operator[](unsigned int index);

    void reset();

    float data[81];
};

//! Complete tensor contraction.
/*! This function is simply a sum-product over two tensors. For reference, see eq. 4.
 *
 *  \param a The first tensor.
 *  \param a The second tensor.
 */ 
float dot(const tensor4& a, const tensor4& b);

//! Generate the r4 tensor.
/*! The r4 tensor is not a word used in the paper, but is a name introduced in
 *  this code to refer to the second term in eqs. 27 in the paper. It is simply
 *  a scaled sum of some delta function products. For convenience, its
 *  calculation is performed in a single function.
 */ 
tensor4 genR4Tensor();


//! Compute the cubatic order parameter for a set of points
/*! The cubatic order parameter is defined according to the paper "Strong
 * orientational coordinates and orientational order parameters for symmetric
 * objects" by Amir Haji-Akbar
 * (http://dx.doi.org/10.1088/1751-8113/48/48/485201). Comments throughout this
 * file reference notes and equations from that paper for clarity. The central
 * idea is to define, for a given symmetry, a minimal set of vectors that can
 * be used to construct a coordinate system that has no degeneracies, i.e. one
 * that is equivalent for any two points that are identical up to any of the
 * transformations in the symmetry group. The set of vectors for a given rigid
 * body R is known as a symmetric descriptor of that object, and is constructed
 * from orbits of that vector (eq. 8). Strong orientational coordinates (SOCs)
 * are then constructed as homogeneous tensors constructed from this set (eq.
 * 3). The central idea of the paper is to then develop tensor functions of the
 * SOCs that can be used to quantify order.
 */
class CubaticOrderParameter
{
public:
    //! Constructor
    CubaticOrderParameter(float t_initial, float t_final, float scale,
                          unsigned int replicates, unsigned int seed);

    //! Destructor
    ~CubaticOrderParameter() {}

    //! Reset the bond order array to all zeros
    void reset();

    //! Compute the cubatic order parameter
    void compute(quat<float>* orientations, unsigned int n);

    //! Get a reference to the last computed cubatic order parameter
    float getCubaticOrderParameter()
    {
        return m_cubatic_order_parameter;
    }

    const util::ManagedArray<float> &getParticleOrderParameter()
    {
        return m_particle_order_parameter;
    }

    std::shared_ptr<float> getParticleTensor()
    {
        return m_particle_tensor;
    }

    const util::ManagedArray<float> &getGlobalTensor()
    {
        m_sp_global_tensor.prepare({3, 3, 3, 3});
        memcpy(m_sp_global_tensor.get(), (void*) &m_global_tensor.data, sizeof(float) * 81);
        return m_sp_global_tensor;
    }

    const util::ManagedArray<float> &getCubaticTensor()
    {
        m_sp_cubatic_tensor.prepare({3, 3, 3, 3});
        memcpy(m_sp_cubatic_tensor.get(), (void*) &m_cubatic_tensor.data, sizeof(float) * 81);
        return m_sp_cubatic_tensor;
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

protected:

    //! Calculate the cubatic tensor
    /*! Implements the second line of eq. 27, the calculation of M_{\omega}.
     *
     *  \param cubatic_tensor The cubatic tensor (denoted M_{\omega} in the paper), overwritten by reference.
     *  \param orientation The orientation that will be used to determine the vectors used in the calculation.
     */
    void calcCubaticTensor(float* cubatic_tensor, quat<float> orientation);

    //! Calculate the scalar cubatic order parameter.
    /*! Implements eq. 22
     *
     *  \param cubatic_order_parameter The output value (updated as a reference)
     *  \param cubatic_tensor The cubatic tensor (denoted M_{\omega} in eq. 22)
     */
    void calcCubaticOrderParameter(float& cubatic_order_parameter, float* cubatic_tensor);

    //! Calculate the per-particle tensor.
    /*! Implements the first line of eq. 27, the calculation of M. The output
     *  is stored in the member variable m_particle_tensor.
     *
     *  \param orientations The per-particle orientations.
     */
    void calculatePerParticleTensor(quat<float>* orientations);

    //! Calculate the global tensor for the system.
    /*! Implements the third line of eq. 27, the calculation of \bar{M}. The output
     *  is stored in the member variable m_global_tensor.
     */
    void calculateGlobalTensor();

    //! Calculate a random quaternion.
    /*! To calculate a random quaternion in a way that obeys the right
     *  distribution of angles, we cannot simply just choose 4 random numbers
     *  and then normalize the quaternion. This function implements an
     *  appropriate calculation.
     */
    quat<float> calcRandomQuaternion(Saru& saru, float angle_multiplier);


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

    util::ManagedArray<float> m_particle_order_parameter; //!< The per-particle value of the order parameter.
    util::ManagedArray<float>
        m_sp_global_tensor; //!< Copy of global tensor used to return persistent data.
    util::ManagedArray<float>
        m_sp_cubatic_tensor; //!< Shared pointer for cubatic tensor, only used to return values to Python.
    std::shared_ptr<float> m_particle_tensor;

    unsigned int m_seed; //!< Random seed

    vec3<float> m_system_vectors[3]; //!< The global coordinate system, always use a simple Euclidean basis.
};

}; }; // end namespace freud::order

#endif // CUBATIC_ORDER_PARAMETER_H
