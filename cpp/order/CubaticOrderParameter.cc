#include "CubaticOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdexcept>
#include <complex>
#include <random>

using namespace std;
using namespace tbb;

/*! \file CubaticOrderParameter.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

CubaticOrderParameter::CubaticOrderParameter(float t_initial, float t_final, float scale, float *r4_tensor)
    : m_t_initial(t_initial), m_t_final(t_final), m_scale(scale), m_n(0), m_n_replicates(1)
    {
    // sanity checks, should be caught in python
    if (m_t_initial < m_t_final)
        throw invalid_argument("t_initial must be greater than t_final");
    if (t_final < 1e-6)
        throw invalid_argument("t_final must be > 1e-6");
    if ((scale > 1) || (scale < 0))
        throw invalid_argument("scale must be between 0 and 1");
    // create tensor arrays
    m_global_tensor = std::shared_ptr<float>(new float[81]);
    m_cubatic_tensor = std::shared_ptr<float>(new float[81]);
    m_particle_tensor = std::shared_ptr<float>(new float[m_n*81]);
    m_particle_order_parameter = std::shared_ptr<float>(new float[m_n]);
    m_gen_r4_tensor = std::shared_ptr<float>(new float[81]);
    memset((void*)m_global_tensor.get(), 0, sizeof(float)*81);
    memset((void*)m_cubatic_tensor.get(), 0, sizeof(float)*81);
    memset((void*)m_particle_tensor.get(), 0, sizeof(float)*m_n*81);
    memset((void*)m_particle_order_parameter.get(), 0, sizeof(float)*m_n);
    // required to not have memory overwritten
    memcpy(m_gen_r4_tensor.get(), r4_tensor, sizeof(float)*81);
    // create random number generators. will be moved to thread specific versions
    m_gen = std::mt19937(m_rd());
    m_theta_dist = std::uniform_real_distribution<float>(0,2.0*M_PI);
    m_phi_dist = std::uniform_real_distribution<float>(0,1.0);
    m_angle_dist = std::uniform_real_distribution<float>(0,2.0*M_PI);
    }

CubaticOrderParameter::~CubaticOrderParameter()
    {
    // for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
    //     {
    //     delete[] (*i);
    //     }
    // for (tbb::enumerable_thread_specific<std::random_device *>::iterator i = m_local_rd.begin(); i != m_local_rd.end(); ++i)
    //     {
    //     delete[] (*i);
    //     }
    // for (tbb::enumerable_thread_specific<std::mt19937 *>::iterator i = m_local_gen.begin(); i != m_local_gen.end(); ++i)
    //     {
    //     delete[] (*i);
    //     }
    // for (tbb::enumerable_thread_specific<std::uniform_real_distribution *>::iterator i = m_local_dist.begin(); i != m_local_dist.end(); ++i)
    //     {
    //     delete[] (*i);
    //     }
    // delete m_nn;
    }

// compute the outer(?) tensor product specific to this problem
// this assumes that an 81 element rank 4 tensor is being computed
// all of these will need to be added to a struct...will make everything better
void tensorProduct(float *tensor, vec3<float> vector)
    {
    // freud does not include an index 4D object, so we're just using a flat array
    unsigned int cnt = 0;
    float v[3];
    v[0] = vector.x;
    v[1] = vector.y;
    v[2] = vector.z;
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
                    tensor[cnt] = v_i * v_j * v_k * v_l;
                    cnt++;
                    }
                }
            }
        }
    }

void tensorMult(float *tensor, float a)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        tensor[i] = tensor[i] * a;
        }
    }

void tensorDiv(float *tensor, float a)
    {
    float a_inv = 1.0/a;
    for (unsigned int i = 0; i < 81; i++)
        {
        tensor[i] = tensor[i] * a_inv;
        }
    }

float tensorDot(float *tensor_a, float *tensor_b)
    {
    // freud does not include an index 4D object, so we're just using a flat array
    float l_sum = 0;
    for (unsigned int i = 0; i < 81; i++)
        {
        l_sum += tensor_a[i] * tensor_b[i];
        }
    return l_sum;
    }

void tensorAdd(float *tensor_out, float *tensor_i, float *tensor_j)
    {
    // freud does not include an index 4D object, so we're just using a flat array
    for (unsigned int i = 0; i < 81; i++)
        {
        tensor_out[i] = tensor_i[i] + tensor_j[i];
        }
    }

void tensorSub(float *tensor_out, float *tensor_i, float *tensor_j)
    {
    // freud does not include an index 4D object, so we're just using a flat array
    for (unsigned int i = 0; i < 81; i++)
        {
        tensor_out[i] = tensor_i[i] - tensor_j[i];
        }
    }

class ComputeParticleTensor
    {
    private:
        float *m_particle_tensor;
        const quat<float> *m_orientations;
        const unsigned int m_n;
    public:
        ComputeParticleTensor(float *particle_tensor,
                            const quat<float> *orientations,
                            const unsigned int n)
            : m_particle_tensor(particle_tensor), m_orientations(orientations), m_n(n)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // create index object to access the array
            Index2D a_i = Index2D(m_n, 81);
            // create the local coordinate system
            vec3<float> v[3];
            v[0] = vec3<float>(1,0,0);
            v[1] = vec3<float>(0,1,0);
            v[2] = vec3<float>(0,0,1);
            // create the tensor object
            float *r4_tensor = new float[81];
            float *l_mbar = new float[81];
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                // get the orientation for the particle
                quat<float> l_orientation = m_orientations[i];
                memset((void*)l_mbar, 0, sizeof(float)*81);
                for (unsigned int j = 0; j < 3; j++)
                    {
                    // set all the values to 0;
                    memset((void*)r4_tensor, 0, sizeof(float)*81);
                    // rotate local vector
                    vec3<float> v_r = rotate(l_orientation, v[j]);
                    tensorProduct(r4_tensor, v_r);
                    tensorAdd(l_mbar, r4_tensor, l_mbar);
                    }
                // apply normalization
                tensorMult(l_mbar,2.0);
                // set the values
                for (unsigned int j = 0; j < 81; j++)
                    {
                    m_particle_tensor[a_i(i,j)] = l_mbar[j];
                    }
                }
            delete r4_tensor;
            delete l_mbar;
            }
    };

class ComputeParticleOrderParameter
    {
    private:
        float *m_particle_order_parameter;
        const float *m_particle_tensor;
        const float *m_cubatic_tensor;
        const unsigned int m_n;
    public:
        ComputeParticleOrderParameter(float *particle_order_parameter,
                                      const float *particle_tensor,
                                      const float *cubatic_tensor,
                                      const unsigned int n)
            : m_particle_order_parameter(particle_order_parameter), m_particle_tensor(particle_tensor),
              m_cubatic_tensor(cubatic_tensor), m_n(n)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // create index object to access the array
            Index2D a_i = Index2D(m_n, 81);
            float diff[81];
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                memset((void*)&diff, 0, sizeof(float)*81);
                tensorSub((float*)&diff, (float*)&m_particle_tensor[a_i(i,0)], (float*)m_cubatic_tensor);
                m_particle_order_parameter[i] = 1.0 - tensorDot((float*)&diff,(float*)&diff)/tensorDot((float*)m_cubatic_tensor,(float*)m_cubatic_tensor);
                }
            }
    };

class ComputeGlobalTensor
    {
    private:
        float *m_m_bar;
        const float *m_particle_tensor;
        const unsigned int m_n;
    public:
        ComputeGlobalTensor(float *m_bar,
                            const float *particle_tensor,
                            const unsigned int n)
            : m_m_bar(m_bar), m_particle_tensor(particle_tensor), m_n(n)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // create index object to access the array
            Index2D a_i = Index2D(m_n, 81);
            float n_inv = 1.0/(float)m_n;
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                float tensor_value = 0;
                for (unsigned int j = 0; j < m_n; j++)
                    {
                    tensor_value += m_particle_tensor[a_i(j,i)];
                    }
                tensor_value *= n_inv;
                m_m_bar[i] = tensor_value;
                }
            }
    };

void CubaticOrderParameter::calcCubaticTensor(float *cubatic_tensor, quat<float> orientation)
    {
    // create the system vectors
    vec3<float> system_vectors[3];
    system_vectors[0] = vec3<float>(1,0,0);
    system_vectors[1] = vec3<float>(0,1,0);
    system_vectors[2] = vec3<float>(0,0,1);
    float calculated_tensor[81];
    memset((void*)&calculated_tensor, 0, sizeof(float)*81);
    // rotate by supplied orientation
    for (unsigned int i=0; i<3; i++)
        {
        system_vectors[i] = rotate(orientation, system_vectors[i]);
        }
    // calculate for each system vector
    for (unsigned int v_idx = 0; v_idx < 3; v_idx++)
        {
        float l_tensor[81];
        tensorProduct((float*)&l_tensor, system_vectors[v_idx]);
        tensorAdd((float*)&calculated_tensor, (float*)&calculated_tensor, (float*)&l_tensor);
        }
    // normalize
    tensorMult((float*)&calculated_tensor, 2.0);
    tensorSub((float*)&calculated_tensor, (float*)&calculated_tensor, m_gen_r4_tensor.get());
    // now, memcpy
    memcpy((void*)cubatic_tensor, &calculated_tensor, sizeof(float)*81);
    }

void CubaticOrderParameter::calcCubaticOrderParameter(float &cubatic_order_parameter, float* cubatic_tensor)
    {
    float diff[81];
    memset((void*)&diff, 0, sizeof(float)*81);
    tensorSub((float*)&diff, m_global_tensor.get(), cubatic_tensor);
    cubatic_order_parameter = 1.0 - tensorDot((float*)&diff,(float*)&diff)/tensorDot(cubatic_tensor,cubatic_tensor);
    }

float CubaticOrderParameter::get_cubatic_order_parameter()
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
    return m_global_tensor;
    }

std::shared_ptr<float> CubaticOrderParameter::getCubaticTensor()
    {
    return m_cubatic_tensor;
    }

std::shared_ptr<float> CubaticOrderParameter::getGenR4Tensor()
    {
    return m_gen_r4_tensor;
    }

unsigned int CubaticOrderParameter::getNumParticles()
    {
    return m_n;
    }

float CubaticOrderParameter::get_t_initial()
    {
    return m_t_initial;
    }

float CubaticOrderParameter::get_t_final()
    {
    return m_t_final;
    }

float CubaticOrderParameter::get_scale()
    {
    return m_scale;
    }

quat<float> CubaticOrderParameter::get_cubatic_orientation()
    {
    return m_cubatic_orientation;
    }

quat<float> CubaticOrderParameter::calcRandomQuaternion(float angle_multiplier=1.0)
    {
    // pull from proper distribution
    float theta = m_theta_dist(m_gen);
    float phi = acos(2.0*m_phi_dist(m_gen)-1.0);
    vec3<float> axis = vec3<float>(cosf(theta)*sinf(phi),sinf(theta)*sinf(phi),cosf(phi));
    float axis_norm = sqrt(dot(axis,axis));
    axis /= axis_norm;
    float angle = angle_multiplier * m_angle_dist(m_gen);
    return quat<float>::fromAxisAngle(axis, angle);
    }

// void CubaticOrderParameter::resetCubaticOrderParameter()
//     {
//     for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
//         {
//         memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);
//         }
//     // reset the frame counter
//     m_frame_counter = 0;
//     }

void CubaticOrderParameter::compute(quat<float> *orientations,
                                    unsigned int n,
                                    unsigned int n_replicates)
    {
    // change the size of the particle tensor if the number of particles
    if (m_n != n)
        {
        m_particle_tensor = std::shared_ptr<float>(new float[n*81]);
        m_particle_order_parameter = std::shared_ptr<float>(new float[n]);
        }
    // reset the values
    memset((void*)m_global_tensor.get(), 0, sizeof(float)*81);
    memset((void*)m_particle_tensor.get(), 0, sizeof(float)*n*81);
    memset((void*)m_particle_order_parameter.get(), 0, sizeof(float)*n);
    // calculate per-particle tensor
    parallel_for(blocked_range<size_t>(0,n),
                 ComputeParticleTensor(m_particle_tensor.get(),
                                       orientations,
                                       n));
    // now calculate the global tensor
    parallel_for(blocked_range<size_t>(0,81),
                 ComputeGlobalTensor(m_global_tensor.get(),
                                     m_particle_tensor.get(),
                                     n));
    // subtract off the general tensor
    // this may not be working...
    Index2D a_i = Index2D(n, 81);
    for (unsigned int i = 0; i < n; i++)
        {
        tensorSub(&m_particle_tensor.get()[a_i(i, 0)], &m_particle_tensor.get()[a_i(i, 0)], m_gen_r4_tensor.get());
        }
    tensorSub(m_global_tensor.get(), m_global_tensor.get(), m_gen_r4_tensor.get());
    // need to generate random orientation
    quat<float> current_orientation = calcRandomQuaternion();
    m_cubatic_orientation = current_orientation;
    // now calculate the cubatic tensor
    calcCubaticTensor(m_cubatic_tensor.get(), current_orientation);
    calcCubaticOrderParameter(m_cubatic_order_parameter, m_cubatic_tensor.get());
    // prep for the simulated annealing
    float t_current = m_t_initial;
    unsigned int loop_count = 0;
    float new_cubatic_tensor[81];
    memset((void*)&new_cubatic_tensor, 0, sizeof(float)*81);
    float new_order_parameter = 0;
    // simulated annealing loop; loop counter to prevent inf loops
    while ((t_current > m_t_final) && (loop_count < 10000))
        {
        loop_count++;
        current_orientation = calcRandomQuaternion(0.1)*m_cubatic_orientation;
        // now calculate the cubatic tensor
        calcCubaticTensor((float*)&new_cubatic_tensor, current_orientation);
        calcCubaticOrderParameter(new_order_parameter, (float*)&new_cubatic_tensor);
        if (new_order_parameter > m_cubatic_order_parameter)
            {
            memcpy(m_cubatic_tensor.get(), (void *)&new_cubatic_tensor, sizeof(float)*81);
            m_cubatic_order_parameter = new_order_parameter;
            m_cubatic_orientation = current_orientation;
            }
        else
            {
            float boltzmann_factor = exp(-(m_cubatic_order_parameter - new_order_parameter) / t_current);
            float test_value = m_phi_dist(m_gen);
            if (boltzmann_factor >= test_value)
                {
                memcpy(m_cubatic_tensor.get(), &new_cubatic_tensor, sizeof(float)*81);
                m_cubatic_order_parameter = new_order_parameter;
                m_cubatic_orientation = current_orientation;
                }
            else
                {
                continue;
                }
            }
        t_current *= m_scale;
        }
    // now calculate the per-particle order parameters
    // this is currently broken...
    parallel_for(blocked_range<size_t>(0,n),
                 ComputeParticleOrderParameter(m_particle_order_parameter.get(),
                                               m_particle_tensor.get(),
                                               m_cubatic_tensor.get(),
                                               n));
    // save the last computed number of particles
    m_n = n;
    m_n_replicates = n_replicates;
    }

}; }; // end namespace freud::order
