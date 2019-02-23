// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <limits>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Eigen/Eigen/Dense"

#include "NematicOrderParameter.h"

using namespace std;
using namespace tbb;

/*! \file NematicOrderParameter.h
    \brief Compute the nematic order parameter for each particle
*/

namespace freud { namespace order {

// m_u is the molecular axis, normalized to a unit vector
NematicOrderParameter::NematicOrderParameter(vec3<float> u)
    : m_n(0), m_u(u/sqrt(dot(u,u))), m_sp_nematic_tensor(
            std::shared_ptr<float>(
                new float [9], std::default_delete<float[]>()))
    {
    }

float NematicOrderParameter::getNematicOrderParameter()
    {
    return m_nematic_order_parameter;
    }

std::shared_ptr<float> NematicOrderParameter::getParticleTensor()
    {
    return m_particle_tensor;
    }

std::shared_ptr<float> NematicOrderParameter::getNematicTensor()
    {
    // return nematic_tensor
    memcpy(m_sp_nematic_tensor.get(), m_nematic_tensor, sizeof(float)*9);
    return m_sp_nematic_tensor;
    }

unsigned int NematicOrderParameter::getNumParticles()
    {
    return m_n;
    }

vec3<float> NematicOrderParameter::getNematicDirector()
    {
    return m_nematic_director;
    }

void NematicOrderParameter::reset()
    {
    memset((void*)m_particle_tensor.get(), 0, sizeof(float)*m_n*9);
    for (unsigned int i = 0; i < 9; ++i)
        m_nematic_tensor[i] = 0.0;
    m_nematic_order_parameter=0.0;
    }

void NematicOrderParameter::compute(quat<float> *orientations,
                                    unsigned int n)
    {
    // change the size of the particle tensor if the number of particles
    if (m_n != n)
        {
        m_particle_tensor = std::shared_ptr<float>(new float[n*9], std::default_delete<float[]>());
        m_n = n;
        }
    // reset the values
    memset((void*)m_particle_tensor.get(), 0, sizeof(float)*n*9);

    // calculate per-particle tensor
    parallel_for(blocked_range<size_t>(0,n),
        [=] (const blocked_range<size_t>& r)
            {
            // create index object to access the array
            Index2D a_i = Index2D(3, 9);

            for (size_t i = r.begin(); i != r.end(); i++)
                {
                // get the director of the particle
                quat<float> q = orientations[i];
                vec3<float> u_i = rotate(q,m_u);

                float Q_ab[9];

                Q_ab[a_i(0,0)] = 1.5f*u_i.x*u_i.x - 0.5f;
                Q_ab[a_i(0,1)] = 1.5f*u_i.x*u_i.y;
                Q_ab[a_i(0,2)] = 1.5f*u_i.x*u_i.z;
                Q_ab[a_i(1,0)] = 1.5f*u_i.y*u_i.x;
                Q_ab[a_i(1,1)] = 1.5f*u_i.y*u_i.y - 0.5f;
                Q_ab[a_i(1,2)] = 1.5f*u_i.y*u_i.z;
                Q_ab[a_i(2,0)] = 1.5f*u_i.z*u_i.x;
                Q_ab[a_i(2,1)] = 1.5f*u_i.z*u_i.y;
                Q_ab[a_i(2,2)] = 1.5f*u_i.z*u_i.z - 0.5f;

                // Set the values. The per-particle array is used so that both
                // this loop and the reduction can be done in parallel afterwards
                for (unsigned int j = 0; j < 9; j++)
                    {
                    m_particle_tensor.get()[i*9+j] += Q_ab[j];
                    }
                }
            });

    // https://stackoverflow.com/questions/9399929/parallel-reduction-of-an-array-on-cpu
    struct reduce_matrix {
        float y_[9];
        const float *m_; // reference to a matrix

        reduce_matrix(const float *m)
            : m_(m)
            {
            for (int i = 0; i < 9; ++i) y_[i] = 0.0; // prepare for accumulation
            }

        // splitting constructor required by TBB
        reduce_matrix(reduce_matrix& rm, tbb::split)
            : m_(rm.m_)
            {
            for (int i = 0; i < 9; ++i) y_[i] = 0.0;
            }

        // adding the elements
        void operator()(const tbb::blocked_range<unsigned int>& r)
            {
            for (unsigned int i = r.begin(); i < r.end(); ++i)
                for (int j = 0; j < 9; ++j)
                    y_[j] += m_[i*9+j];
            }

        // reduce computations in two matrices
        void join(reduce_matrix& rm)
            {
            for (int i = 0; i < 9; ++i) y_[i] += rm.y_[i];
            }
        };

    // now calculate the sum of Q_ab's
    reduce_matrix matrix(m_particle_tensor.get());

    parallel_reduce(blocked_range<unsigned int>(0,m_n), matrix);

    // set the averaged Q_ab
    for (unsigned int i = 0; i < 9; ++i)
        m_nematic_tensor[i] = matrix.y_[i]/m_n;

    // get the nematic order parameter and director
    Eigen::MatrixXf m(3,3);
    Index2D a_i = Index2D(3, 3);
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            {
            m(i,j) = m_nematic_tensor[a_i(i,j)];
            }

    Eigen::EigenSolver<Eigen::MatrixXf> es;
    es.compute(m);

    float evec[9];
    float eval[3];
    if (es.info() != Eigen::Success)
        {
        // numerical issue, set r to identity matrix
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                if (i == j)
                    {
                    evec[a_i(i,j)] = 1.0;
                    }
                else
                    {
                    evec[a_i(j,j)] = 0.0;
                    }
        // set order parameter to zero so it's easily detectable
        eval[0] = eval[1] = eval[2] = 0.0;
        }
    else
        {
        // columns are eigenvectors
        Eigen::MatrixXcf eigen_vec = es.eigenvectors();
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                evec[a_i(i,j)] = eigen_vec(i,j).real();
        auto eigen_val = es.eigenvalues();
        eval[0] = eigen_val(0).real();
        eval[1] = eigen_val(1).real();
        eval[2] = eigen_val(2).real();
        }

    // the order parameter is the eigenvector belonging to the largest eigenvalue
    unsigned int max_idx = 0;
    float max_val = std::numeric_limits<float>::lowest();

    for (unsigned int i = 0; i < 3; ++i)
        if (eval[i] > max_val)
            {
            max_val = eval[i];
            max_idx = i;
            }

    m_nematic_director = vec3<Scalar>(evec[a_i(0,max_idx)],evec[a_i(1,max_idx)],evec[a_i(2,max_idx)]);
    m_nematic_order_parameter = max_val;
    }

}; }; // end namespace freud::order
