// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "CubaticOrderParameter.h"

using namespace std;
using namespace tbb;

/*! \file CubaticOrderParameter.h
    \brief Compute the cubatic order parameter for each particle.
*/

namespace freud { namespace order {

CubaticOrderParameter::CubaticOrderParameter(float t_initial, float t_final, float scale, float *r4_tensor,
    unsigned int replicates, unsigned int seed)
    : m_t_initial(t_initial), m_t_final(t_final), m_scale(scale), m_n(0), m_replicates(replicates), m_seed(seed)
    {
    // sanity checks, should be caught in python
    if (m_t_initial < m_t_final)
        throw invalid_argument("CubaticOrderParameter requires that t_initial must be greater than t_final.");
    if (t_final < 1e-6)
        throw invalid_argument("CubaticOrderParameter requires that t_final must be >= 1e-6.");
    if ((scale > 1) || (scale < 0))
        throw invalid_argument("CubaticOrderParameter requires that scale must be between 0 and 1.");

    // required to not have memory overwritten
    memset((void*)&m_global_tensor.data, 0, sizeof(float)*81);
    memset((void*)&m_cubatic_tensor.data, 0, sizeof(float)*81);
    memcpy((void*)&m_gen_r4_tensor.data, r4_tensor, sizeof(float)*81);

    // Create shared pointer tensor arrays, which are used for returning to Python.
    m_particle_tensor = std::shared_ptr<float>(new float[m_n*81], std::default_delete<float[]>());
    m_particle_order_parameter = std::shared_ptr<float>(new float[m_n], std::default_delete<float[]>());
    m_sp_global_tensor = std::shared_ptr<float>(new float[81], std::default_delete<float[]>());
    m_sp_cubatic_tensor = std::shared_ptr<float>(new float[81], std::default_delete<float[]>());
    m_sp_gen_r4_tensor = std::shared_ptr<float>(new float[81], std::default_delete<float[]>());

    // Initialize the shared pointers
    memset((void*)m_particle_tensor.get(), 0, sizeof(float)*m_n*81);
    memset((void*)m_particle_order_parameter.get(), 0, sizeof(float)*m_n);
    memset((void*)m_sp_global_tensor.get(), 0, sizeof(float)*m_n*81);
    memset((void*)m_sp_cubatic_tensor.get(), 0, sizeof(float)*m_n*81);
    memset((void*)m_sp_gen_r4_tensor.get(), 0, sizeof(float)*m_n*81);

    // create random number generator.
    Saru m_saru(m_seed, 0, 0xffaabb);
    }

void CubaticOrderParameter::calcCubaticTensor(float *cubatic_tensor, quat<float> orientation)
    {
    // create the system vectors
    vec3<float> system_vectors[3];
    system_vectors[0] = vec3<float>(1,0,0);
    system_vectors[1] = vec3<float>(0,1,0);
    system_vectors[2] = vec3<float>(0,0,1);
    tensor4<float> calculated_tensor = tensor4<float>();
    // rotate by supplied orientation
    for (unsigned int i=0; i<3; i++)
        {
        system_vectors[i] = rotate(orientation, system_vectors[i]);
        }
    // calculate for each system vector
    for (unsigned int v_idx = 0; v_idx < 3; v_idx++)
        {
        tensor4<float> l_tensor(system_vectors[v_idx]);
        calculated_tensor += l_tensor;
        }
    // normalize
    calculated_tensor *= (float) 2.0;
    calculated_tensor -= m_gen_r4_tensor;
    // now, memcpy
    memcpy((void*)cubatic_tensor, (void*)&calculated_tensor.data, sizeof(float)*81);
    }

void CubaticOrderParameter::calcCubaticOrderParameter(float &cubatic_order_parameter, float* cubatic_tensor)
    {
    tensor4<float> l_cubatic_tensor = tensor4<float>();
    memcpy((void*)&l_cubatic_tensor.data, (void*)cubatic_tensor, sizeof(float)*81);
    tensor4<float> diff;
    diff = m_global_tensor - l_cubatic_tensor;
    cubatic_order_parameter = 1.0 - dot(diff, diff)/dot(l_cubatic_tensor, l_cubatic_tensor);
    }

float CubaticOrderParameter::getCubaticOrderParameter()
    {
    return m_cubatic_order_parameter;
    }

std::shared_ptr<float> CubaticOrderParameter::getParticleCubaticOrderParameter()
    {
    return m_particle_order_parameter;
    }

std::shared_ptr<float> CubaticOrderParameter::getParticleTensor()
    {
    return m_particle_tensor;
    }

std::shared_ptr<float> CubaticOrderParameter::getGlobalTensor()
    {
    memcpy(m_sp_global_tensor.get(), (void*)&m_global_tensor.data, sizeof(float)*81);
    return m_sp_global_tensor;
    }

std::shared_ptr<float> CubaticOrderParameter::getCubaticTensor()
    {
    memcpy(m_sp_cubatic_tensor.get(), (void*)&m_cubatic_tensor.data, sizeof(float)*81);
    return m_sp_cubatic_tensor;
    }

std::shared_ptr<float> CubaticOrderParameter::getGenR4Tensor()
    {
    memcpy(m_sp_gen_r4_tensor.get(), (void*)&m_gen_r4_tensor.data, sizeof(float)*81);
    return m_sp_gen_r4_tensor;
    }

unsigned int CubaticOrderParameter::getNumParticles()
    {
    return m_n;
    }

float CubaticOrderParameter::getTInitial()
    {
    return m_t_initial;
    }

float CubaticOrderParameter::getTFinal()
    {
    return m_t_final;
    }

float CubaticOrderParameter::getScale()
    {
    return m_scale;
    }

quat<float> CubaticOrderParameter::getCubaticOrientation()
    {
    return m_cubatic_orientation;
    }

quat<float> CubaticOrderParameter::calcRandomQuaternion(Saru &saru, float angle_multiplier=1.0)
    {
    // pull from proper distribution
    float theta = saru.s<float>(0,2.0*M_PI);
    float phi = acos(2.0*saru.s<float>(0,1)-1.0);
    vec3<float> axis = vec3<float>(cosf(theta)*sinf(phi),sinf(theta)*sinf(phi),cosf(phi));
    float axis_norm = sqrt(dot(axis,axis));
    axis /= axis_norm;
    float angle = angle_multiplier * saru.s<float>(0,1);
    return quat<float>::fromAxisAngle(axis, angle);
    }

void CubaticOrderParameter::compute(quat<float> *orientations,
                                    unsigned int n,
                                    unsigned int replicates)
    {
    // change the size of the particle tensor if the number of particles
    if (m_n != n)
        {
        m_particle_tensor = std::shared_ptr<float>(new float[n*81], std::default_delete<float[]>());
        m_particle_order_parameter = std::shared_ptr<float>(new float[n], std::default_delete<float[]>());
        }
    // reset the values
    memset((void*)&m_global_tensor.data, 0, sizeof(float)*81);
    memset((void*)m_particle_tensor.get(), 0, sizeof(float)*n*81);
    memset((void*)m_particle_order_parameter.get(), 0, sizeof(float)*n);
    // calculate per-particle tensor
    parallel_for(blocked_range<size_t>(0,n),
        [=] (const blocked_range<size_t>& r)
            {
            // create index object to access the array
            Index2D a_i = Index2D(n, 81);
            // create the local coordinate system
            vec3<float> v[3];
            v[0] = vec3<float>(1,0,0);
            v[1] = vec3<float>(0,1,0);
            v[2] = vec3<float>(0,0,1);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                // get the orientation for the particle
                quat<float> l_orientation = orientations[i];
                tensor4<float> l_mbar = tensor4<float>();
                for (unsigned int j = 0; j < 3; j++)
                    {
                    // rotate local vector
                    vec3<float> v_r = rotate(l_orientation, v[j]);
                    tensor4<float> r4_tensor(v_r);
                    l_mbar += r4_tensor;
                    }
                // apply normalization
                l_mbar *= (float)2.0;
                // set the values
                for (unsigned int j = 0; j < 81; j++)
                    {
                    m_particle_tensor.get()[a_i(i,j)] = l_mbar.data[j];
                    }
                }
            });
    // now calculate the global tensor
    parallel_for(blocked_range<size_t>(0,81),
        [=] (const blocked_range<size_t>& r)
           {
           // create index object to access the array
           Index2D a_i = Index2D(n, 81);
           float n_inv = 1.0/(float)n;
           for (size_t i = r.begin(); i != r.end(); i++)
               {
               float tensor_value = 0;
               for (unsigned int j = 0; j < n; j++)
                   {
                   tensor_value += m_particle_tensor.get()[a_i(j,i)];
                   }
               tensor_value *= n_inv;
               m_global_tensor.data[i] = tensor_value;
               }
           });
    // subtract off the general tensor
    m_global_tensor -= m_gen_r4_tensor;
    // prep for the simulated annealing
    std::shared_ptr<float> p_cubatic_tensor = std::shared_ptr<float>(new float[m_replicates*81], std::default_delete<float[]>());
    memset((void*)p_cubatic_tensor.get(), 0, sizeof(float)*m_replicates*81);
    std::shared_ptr<float> p_cubatic_order_parameter = std::shared_ptr<float>(new float[m_replicates], std::default_delete<float[]>());
    memset((void*)p_cubatic_order_parameter.get(), 0, sizeof(float)*m_replicates);
    std::shared_ptr< quat<float> > p_cubatic_orientation = std::shared_ptr< quat<float> >(new quat<float>[m_replicates], std::default_delete< quat<float>[]>());
    memset((void*)p_cubatic_orientation.get(), 0, sizeof(quat<float>)*m_replicates);
    // parallel for to handle the replicates...
    parallel_for(blocked_range<size_t>(0, m_replicates),
        [=] (const blocked_range<size_t>& r)
            {
            // create thread-specific rng
            unsigned int thread_start = (unsigned int)r.begin();
            Saru l_saru(m_seed, thread_start, 0xffaabb);
            // create Index2D to access shared arrays
            Index2D a_i = Index2D(m_replicates, 81);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                tensor4<float> cubatic_tensor;
                tensor4<float> new_cubatic_tensor;
                // need to generate random orientation
                quat<float> cubatic_orientation = calcRandomQuaternion(l_saru);
                quat<float> current_orientation = cubatic_orientation;
                float cubatic_order_parameter = 0;
                // now calculate the cubatic tensor
                calcCubaticTensor((float*)&cubatic_tensor.data, cubatic_orientation);
                calcCubaticOrderParameter(cubatic_order_parameter, (float*)&cubatic_tensor.data);
                float new_order_parameter = cubatic_order_parameter;
                // set initial temperature and count
                float t_current = m_t_initial;
                unsigned int loop_count = 0;
                // simulated annealing loop; loop counter to prevent inf loops
                while ((t_current > m_t_final) && (loop_count < 10000))
                    {
                    loop_count++;
                    current_orientation = calcRandomQuaternion(l_saru, 0.1)*(cubatic_orientation);
                    // now calculate the cubatic tensor
                    calcCubaticTensor((float*)&new_cubatic_tensor.data, current_orientation);
                    calcCubaticOrderParameter(new_order_parameter, (float*)&new_cubatic_tensor.data);
                    if (new_order_parameter > cubatic_order_parameter)
                        {
                        memcpy((void*)&cubatic_tensor.data, (void*)&new_cubatic_tensor.data, sizeof(float)*81);
                        cubatic_order_parameter = new_order_parameter;
                        cubatic_orientation = current_orientation;
                        }
                    else
                        {
                        float boltzmann_factor = exp(-(cubatic_order_parameter - new_order_parameter) / t_current);
                        float test_value = l_saru.s<float>(0,1);
                        if (boltzmann_factor >= test_value)
                            {
                            memcpy((void*)&cubatic_tensor.data, (void*)&new_cubatic_tensor.data, sizeof(float)*81);
                            cubatic_order_parameter = new_order_parameter;
                            cubatic_orientation = current_orientation;
                            }
                        else
                            {
                            continue;
                            }
                        }
                    t_current *= m_scale;
                    }
                // set values
                float *tensor_ptr = (float*)&(p_cubatic_tensor.get()[a_i(i,0)]);
                memcpy((void*)tensor_ptr, (void*)&cubatic_tensor, sizeof(float)*81);
                p_cubatic_orientation.get()[i].s = cubatic_orientation.s;
                p_cubatic_orientation.get()[i].v = cubatic_orientation.v;
                p_cubatic_order_parameter.get()[i] = cubatic_order_parameter;
                }
            });
    // now, find max and set the values
    unsigned int max_idx = 0;
    float max_cubatic_order_parameter = p_cubatic_order_parameter.get()[max_idx];
    for (unsigned int i = 1; i < m_replicates; i++)
        {
        if (p_cubatic_order_parameter.get()[i] > max_cubatic_order_parameter)
            {
            max_idx = i;
            max_cubatic_order_parameter = p_cubatic_order_parameter.get()[i];
            }
        }
    // set the values
    Index2D a_i = Index2D(m_replicates, 81);
    memcpy((void*)&m_cubatic_tensor.data, (void*)&(p_cubatic_tensor.get()[a_i(max_idx,0)]), sizeof(float)*81);
    m_cubatic_orientation.s = p_cubatic_orientation.get()[max_idx].s;
    m_cubatic_orientation.v = p_cubatic_orientation.get()[max_idx].v;
    m_cubatic_order_parameter = p_cubatic_order_parameter.get()[max_idx];
    // now calculate the per-particle order parameters
    parallel_for(blocked_range<size_t>(0,n),
        [=] (const blocked_range<size_t>& r)
            {
            tensor4<float> l_mbar;
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                // use the cubatic OP calc to compute per-particle OP
                // i.e. what is the value of the COP
                // if the global orientation were the particle orientation
                // load the orientation
                tensor4<float> l_particle_tensor;
                float l_particle_op;
                quat<float> l_orientation = orientations[i];
                calcCubaticTensor((float*)&l_particle_tensor.data, l_orientation);
                calcCubaticOrderParameter(l_particle_op, (float*)&l_particle_tensor.data);
                m_particle_order_parameter.get()[i] = l_particle_op;
                }
            });
    // save the last computed number of particles
    m_n = n;
    m_replicates = replicates;
    }

}; }; // end namespace freud::order
