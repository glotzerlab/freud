// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <iostream>

#include "CubaticOrderParameter.h"
#include "Index1D.h"

using namespace std;
using namespace tbb;

/*! \file CubaticOrderParameter.h
    \brief Compute the cubatic order parameter for each particle.
*/

namespace freud { namespace order {


tensor4::tensor4()
{
    memset((void*) &data, 0, sizeof(float) * 81);
}
tensor4::tensor4(vec3<float> vector)
{
    unsigned int cnt = 0;
    float v[3];
    v[0] = vector.x;
    v[1] = vector.y;
    v[2] = vector.z;
    for (unsigned int i = 0; i < 3; i++)
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            for (unsigned int k = 0; k < 3; k++)
            {
                for (unsigned int l = 0; l < 3; l++)
                {
                    data[cnt] = v[i] * v[j] * v[k] * v[l];
                    cnt++;
                }
            }
        }
    }
}

//! Writeable index into array.
float &tensor4::operator[](unsigned int index)
{
    return data[index];
}

tensor4 tensor4::operator+=(const tensor4& b)
{
    for (unsigned int i = 0; i < 81; i++)
    {
        data[i] += b.data[i];
    }
    return *this;
}

tensor4 tensor4::operator+=(const float& b)
{
    for (unsigned int i = 0; i < 81; i++)
    {
        data[i] += b;
    }
    return *this;
}

tensor4 tensor4::operator-(const tensor4& b)
{
    tensor4 c;
    for (unsigned int i = 0; i < 81; i++)
    {
        c.data[i] = data[i] - b.data[i];
    }
    return c;
}

tensor4 tensor4::operator-=(const tensor4& b)
{
    for (unsigned int i = 0; i < 81; i++)
    {
        data[i] -= b.data[i];
    }
    return *this;
}

tensor4 tensor4::operator*(const float& b)
{
    tensor4 c;
    for (unsigned int i = 0; i < 81; i++)
    {
        c.data[i] = data[i] * b;
    }
    return c;
}

tensor4 tensor4::operator*=(const float& b)
{
    for (unsigned int i = 0; i < 81; i++)
    {
        data[i] *= b;
    }
    return *this;
}

void tensor4::reset()
{
    memset((void *) &data, 0, sizeof(float) * 81);
}

void tensor4::copyToManagedArray(util::ManagedArray<float> &ma)
{
    memcpy(ma.get(), (void*) &data, sizeof(float) * 81);
}

float dot(const tensor4& a, const tensor4& b)
{
    float c = 0;
    for (unsigned int i = 0; i < 81; i++)
    {
        c += a.data[i] * b.data[i];
    }
    return c;
}


tensor4 genR4Tensor()
{
    // Construct the identity matrix to build the delta functions.
    util::ManagedArray<float> identity({3, 3});
    identity(0, 0) = 1;
    identity(1, 1) = 1;
    identity(2, 2) = 1;

    unsigned int cnt = 0;
    tensor4 r4 = tensor4();
    r4.reset();
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                {
                    // ijkl term
                    r4[cnt] += identity(i, j)*identity(k, l);
                    // ikjl term
                    r4[cnt] += identity(i, k)*identity(j, l);
                    // iljk term
                    r4[cnt] += identity(i, l)*identity(j, k);
                    r4[cnt] *= 2.0/5.0;
                    ++cnt;
                }
    return r4;
}


CubaticOrderParameter::CubaticOrderParameter(float t_initial, float t_final, float scale,
                                             unsigned int replicates, unsigned int seed)
    : m_t_initial(t_initial), m_t_final(t_final), m_scale(scale), m_n(0), m_replicates(replicates),
      m_seed(seed)
{
    // sanity checks, should be caught in python
    if (m_t_initial < m_t_final)
        throw invalid_argument("CubaticOrderParameter requires that t_initial must be greater than t_final.");
    if (t_final < 1e-6)
        throw invalid_argument("CubaticOrderParameter requires that t_final must be >= 1e-6.");
    if ((scale > 1) || (scale < 0))
        throw invalid_argument("CubaticOrderParameter requires that scale must be between 0 and 1.");

    // required to not have memory overwritten
    m_gen_r4_tensor = genR4Tensor();

    // Initialize the system vectors using Euclidean vectors.
    m_system_vectors[0] = vec3<float>(1, 0, 0);
    m_system_vectors[1] = vec3<float>(0, 1, 0);
    m_system_vectors[2] = vec3<float>(0, 0, 1);
}

tensor4 CubaticOrderParameter::calcCubaticTensor(quat<float> orientation)
{
    tensor4 calculated_tensor = tensor4();

    // The cubatic tensor is computed by rotating each basis vector by the
    // provided rotation and then summing the resulting tensors.
    for (unsigned int i = 0; i < 3; i++)
    {
        calculated_tensor += tensor4(rotate(orientation, m_system_vectors[i]));
    }

    // normalize
    calculated_tensor *= (float) 2.0;
    calculated_tensor -= m_gen_r4_tensor;
    return calculated_tensor;
}

float CubaticOrderParameter::calcCubaticOrderParameter(tensor4 cubatic_tensor, tensor4 global_tensor)
{
    tensor4 diff = global_tensor - cubatic_tensor;
    return 1.0 - dot(diff, diff) / dot(cubatic_tensor, cubatic_tensor);
}

quat<float> CubaticOrderParameter::calcRandomQuaternion(Saru& saru, float angle_multiplier = 1.0)
{
    float theta = saru.s<float>(0, 2.0 * M_PI);
    float phi = acos(2.0 * saru.s<float>(0, 1) - 1.0);
    vec3<float> axis = vec3<float>(cosf(theta) * sinf(phi), sinf(theta) * sinf(phi), cosf(phi));
    float axis_norm = sqrt(dot(axis, axis));
    axis /= axis_norm;
    float angle = angle_multiplier * saru.s<float>(0, 1);
    return quat<float>::fromAxisAngle(axis, angle);
}

void CubaticOrderParameter::calculatePerParticleTensor(quat<float>* orientations)
{
    // calculate per-particle tensor
    parallel_for(blocked_range<size_t>(0, m_n), [=](const blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            tensor4 l_mbar = tensor4();
            for (unsigned int j = 0; j < 3; ++j)
            {
                // Calculate the homogeneous tensor H for each vector then add
                // to the per-particle value.
                vec3<float> v_r = rotate(orientations[i], m_system_vectors[j]);
                tensor4 r4_tensor(v_r);
                l_mbar += r4_tensor;
            }

            // Apply the prefactor from the sum in equation 27 before assigning.
            m_particle_tensor[i] = l_mbar*2.0;
        }
    });
}

tensor4 CubaticOrderParameter::calculateGlobalTensor()
{
    tensor4 global_tensor;
    global_tensor.reset();

    // now calculate the global tensor
    parallel_for(blocked_range<size_t>(0, 81), [=, &global_tensor](const blocked_range<size_t>& r) {
        // create index object to access the array
        float n_inv = 1.0 / (float) m_n;
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            float tensor_value = 0;
            for (unsigned int j = 0; j < m_n; j++)
            {
                tensor_value += m_particle_tensor[j][i];
            }
            // Note that in the third equation in eq. 27, the prefactor of the
            // sum is 2/N, but the factor of 2 is already accounted for in the
            // calculation of per particle calculation in
            // calculatePerParticleTensor.
            tensor_value *= n_inv;
            global_tensor.data[i] = tensor_value;
        }
    });
    // Subtract off the general tensor
    global_tensor -= m_gen_r4_tensor;

    return global_tensor;
}

void CubaticOrderParameter::compute(quat<float>* orientations, unsigned int n)
{
    m_n = n;
    m_particle_tensor.prepare(m_n);
    m_particle_order_parameter.prepare(m_n);

    // Calculate the per-particle tensor
    calculatePerParticleTensor(orientations);
    tensor4 global_tensor = calculateGlobalTensor();
    m_global_tensor.prepare({3, 3, 3, 3});
    global_tensor.copyToManagedArray(m_global_tensor);

    // The paper recommends using a Newton-Raphson scheme to optimize the order
    // parameter, but in practice we find that simulated annealing performs
    // much better, so we perform replicates of the process and choose the best
    // one.
    util::ManagedArray<tensor4> p_cubatic_tensor({m_replicates});
    util::ManagedArray<float> p_cubatic_order_parameter({m_replicates});
    util::ManagedArray<quat<float> > p_cubatic_orientation({m_replicates});

    parallel_for(blocked_range<size_t>(0, m_replicates), [=, &p_cubatic_orientation, &p_cubatic_order_parameter, &p_cubatic_tensor](const blocked_range<size_t>& r) {

        // create thread-specific rng
        unsigned int thread_start = (unsigned int) r.begin();
        Saru l_saru(m_seed, thread_start, 0xffaabb);

        // create Index2D to access shared arrays
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            tensor4 cubatic_tensor;
            tensor4 new_cubatic_tensor;

            // need to generate random orientation
            quat<float> cubatic_orientation = calcRandomQuaternion(l_saru);
            quat<float> new_orientation = cubatic_orientation;

            // now calculate the cubatic tensor
            cubatic_tensor = calcCubaticTensor(cubatic_orientation);
            float cubatic_order_parameter = calcCubaticOrderParameter(cubatic_tensor, global_tensor);
            float new_order_parameter = cubatic_order_parameter;

            // set initial temperature and count
            float t_current = m_t_initial;
            unsigned int loop_count = 0;
            // simulated annealing loop; loop counter to prevent inf loops
            while ((t_current > m_t_final) && (loop_count < 10000))
            {
                loop_count++;
                new_orientation = calcRandomQuaternion(l_saru, 0.1) * (cubatic_orientation);
                // now calculate the cubatic tensor
                new_cubatic_tensor = calcCubaticTensor(new_orientation);
                new_order_parameter = calcCubaticOrderParameter(new_cubatic_tensor, global_tensor);
                if (new_order_parameter > cubatic_order_parameter)
                {
                    cubatic_tensor = new_cubatic_tensor;
                    cubatic_order_parameter = new_order_parameter;
                    cubatic_orientation = new_orientation;
                }
                else
                {
                    float boltzmann_factor
                        = exp(-(cubatic_order_parameter - new_order_parameter) / t_current);
                    if (boltzmann_factor >= l_saru.s<float>(0, 1))
                    {
                        cubatic_tensor = new_cubatic_tensor;
                        cubatic_order_parameter = new_order_parameter;
                        cubatic_orientation = new_orientation;
                    }
                    else
                    {
                        continue;
                    }
                }
                t_current *= m_scale;
            }
            // set values
            p_cubatic_tensor[i] = cubatic_tensor;
            p_cubatic_orientation[i].s = cubatic_orientation.s;
            p_cubatic_orientation[i].v = cubatic_orientation.v;
            p_cubatic_order_parameter[i] = cubatic_order_parameter;
        }
    });

    // Loop over threads and choose the replicate that found the highest order.
    unsigned int max_idx = 0;
    float max_cubatic_order_parameter = p_cubatic_order_parameter[max_idx];
    for (unsigned int i = 1; i < m_replicates; ++i)
    {
        if (p_cubatic_order_parameter[i] > max_cubatic_order_parameter)
        {
            max_idx = i;
            max_cubatic_order_parameter = p_cubatic_order_parameter[i];
        }
    }

    // set the values
    m_cubatic_tensor.prepare({3, 3, 3, 3});
    p_cubatic_tensor[max_idx].copyToManagedArray(m_cubatic_tensor);
    m_cubatic_orientation.s = p_cubatic_orientation[max_idx].s;
    m_cubatic_orientation.v = p_cubatic_orientation[max_idx].v;
    m_cubatic_order_parameter = p_cubatic_order_parameter[max_idx];

    // now calculate the per-particle order parameters
    parallel_for(blocked_range<size_t>(0, m_n), [=](const blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            // use the cubatic OP calc to compute per-particle OP
            // i.e. what is the value of the COP
            // if the global orientation were the particle orientation
            // load the orientation
            m_particle_order_parameter[i] = calcCubaticOrderParameter(calcCubaticTensor(orientations[i]), global_tensor);;
        }
    });
}

}; }; // end namespace freud::order
