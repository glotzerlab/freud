#include "CubaticOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdlib.h>
#include <math.h>
#include <stdexcept>
#include <complex>
#include <tbb/tbb.h>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include <ctime>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

using namespace std;
using namespace tbb;

/*! \file CubaticOrderParameter.cc
    \brief Compute the global cubatic order parameter
*/

namespace freud { namespace order {

CubaticOrderParameter::CubaticOrderParameter(float tInitial, float tFinal, float scale, float norm)
    : m_box(trajectory::Box()), m_tInitial(tInitial), m_tFinal(tFinal), m_scale(scale), m_norm(norm), m_Np(0), m_rng((unsigned int)std::time(0)), m_rngDist(0,1), m_rngGen(m_rng, m_rngDist)
    {
    m_p4Sum = 0.0;
    m_p4SumNew = 0.0;
    // create the array to carry the OPs
    m_p4_array = boost::shared_array<float>(new float[3]);
    }

CubaticOrderParameter::~CubaticOrderParameter()
    {
    }

class ComputeCubaticOrderParameter
    {
    private:
        atomic<float> &m_p4Sum;
        const quat<float> m_trial;
        const quat<float> *m_orientations;
    public:
        ComputeCubaticOrderParameter(atomic<float> &p4Sum,
                                 const quat<float> trial,
                                 const quat<float> *orientations)
            : m_p4Sum(p4Sum), m_trial(trial), m_orientations(orientations)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float l_sum = 0.0;
            quat<float> trial = m_trial;

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                quat<float> ref = m_orientations[i];
                // float angle = acos(2.0 * powf(ref.s*trial.s + ref.v.x*trial.v.x + ref.v.y*trial.v.y + ref.v.z*trial.v.z, 2.0) - 1.0);
                // new idea
                quat<float> q12 = trial * conj(ref);
                float angle = 2.0 * acos(q12.s);
                float p4_i = 35.0*powf(cosf(angle), 4) - 30.0*powf(cosf(angle), 2) + 3.0;
                // printf("partial sum = %f", p4_i);
                l_sum += p4_i;
                }
            m_p4Sum = m_p4Sum + l_sum;
            }
    };

void CubaticOrderParameter::compute(trajectory::Box& box,
                                    const quat<float> *orientations,
                                    unsigned int Np)
    {

    m_box = box;

    // new method
    //
    // create the initial vectors
    vec3<float> v0(1.0,0.0,0.0);
    vec3<float> v1(0.0,1.0,0.0);
    vec3<float> v2(0.0,0.0,1.0);
    // create the quaternions which take 0->1, 0->2;
    quat<float> q01 = quat<float>::fromAxisAngle(vec3<float>(0.0,0.0,1.0),M_PI/2.0);
    quat<float> q02 = quat<float>::fromAxisAngle(vec3<float>(0.0,1.0,0.0),-M_PI/2.0);
    // below code verifies, doesn't need to run
    // vec3<float> v01 = rotate(q01, v0);
    // vec3<float> v02 = rotate(q02, v0);
    // printf("v1 = %f %f %f\n", v1.x, v1.y, v1.z);
    // printf("v01 = %f %f %f\n", v01.x, v01.y, v01.z);
    // printf("v2 = %f %f %f\n", v2.x, v2.y, v2.z);
    // printf("v02 = %f %f %f\n", v02.x, v02.y, v02.z);

    // find q0 via simulated annealing
    // generate first trial rotation and normalize
    // http://mathworld.wolfram.com/SpherePointPicking.html
    float theta(2.0 * M_PI * m_rngGen());
    float phi(acos(2.0 * m_rngGen() - 1));
    vec3<float> axis(cosf(theta) * sinf(phi), sinf(theta) * sinf(phi), cos(phi));
    axis /= sqrt(dot(axis, axis));
    float angle(2.0 * M_PI * m_rngGen());
    // generate the quaternion
    m_trial = quat<float>::fromAxisAngle(axis, angle);
    // initialize the sum to zero
    m_p4Sum = 0.0;
    // calculate the sum for the current trial vector
    parallel_for(blocked_range<size_t>(0,Np),
                 ComputeCubaticOrderParameter(m_p4Sum,
                                              m_trial,
                                              orientations));
    // normalize the sum
    m_p4Sum = m_p4Sum * (m_norm / Np);

    m_tCurrent = m_tInitial;
    printf("first simulated annealing\n");
    while (m_tCurrent > m_tFinal)
        {
        // perform the simulated annealing
        bool keepAnnealing = true;
        while (keepAnnealing)
            {
            // generate trial rotation of the global orientation
            // pick axis at random
            theta = 2.0 * M_PI * m_rngGen();
            phi = acos(2.0 * m_rngGen() - 1);
            axis.x = cosf(theta) * sinf(phi);
            axis.y = sinf(theta) * sinf(phi);
            axis.z = cos(phi);
            axis /= sqrt(dot(axis, axis));
            // pick an angle to rotate by, scaling by temperature
            // arbitrary scaling factor, can be changed, should be user-set
            angle = 10.0 * m_tCurrent * M_PI * m_rngGen();
            quat<float> newTrial = quat<float>::fromAxisAngle(axis, angle) * m_trial;
            m_p4SumNew = 0.0;
            // calculate the sum for the new trial vector
            parallel_for(blocked_range<size_t>(0,Np),
                         ComputeCubaticOrderParameter(m_p4SumNew,
                                                      newTrial,
                                                      orientations));
            // normalize the sum
            m_p4SumNew = m_p4SumNew * (m_norm / Np);
            // boltzmann criterion
            if (m_p4SumNew >= m_p4Sum)
                {
                m_trial = newTrial;
                m_p4Sum = m_p4SumNew;
                keepAnnealing = false;
                }
            else
                {
                float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
                float test = m_rngGen();
                if (factor >= test)
                    {
                    m_trial = newTrial;
                    m_p4Sum = m_p4SumNew;
                    keepAnnealing = false;
                    }
                }
            }
        // decrease temperature
        m_tCurrent *= m_scale;
        }
    m_cq0 = m_trial;
    m_p4Sum0 = m_p4Sum;
    // now we have one orientation, we can generate the other two from our knowledge of the
    // cubatic symmetry
    m_cq1 = m_cq0 * q01;
    m_cq2 = m_cq0 * q02;
    // calc the sums
    // initialize the sum to zero
    m_p4Sum = 0.0;
    // calculate the sum for the current trial vector
    parallel_for(blocked_range<size_t>(0,Np),
                 ComputeCubaticOrderParameter(m_p4Sum,
                                              m_cq1,
                                              orientations));
    // normalize the sum
    m_p4Sum = m_p4Sum * (m_norm / Np);
    m_p4Sum1 = m_p4Sum;
    // initialize the sum to zero
    m_p4Sum = 0.0;
    // calculate the sum for the current trial vector
    parallel_for(blocked_range<size_t>(0,Np),
                 ComputeCubaticOrderParameter(m_p4Sum,
                                              m_cq2,
                                              orientations));
    // normalize the sum
    m_p4Sum = m_p4Sum * (m_norm / Np);
    m_p4Sum2 = m_p4Sum;

    // now we refine by optimizing the RMS
    float m_RMS = sqrt((m_p4Sum0*m_p4Sum0 + m_p4Sum1*m_p4Sum1 + m_p4Sum2*m_p4Sum2)/3.0);
    // initialize local sums to 0
    tbb::atomic<float> l_p4SumNew0; l_p4SumNew0 = 0.0;
    tbb::atomic<float> l_p4SumNew1; l_p4SumNew1 = 0.0;
    tbb::atomic<float> l_p4SumNew2; l_p4SumNew2 = 0.0;

    printf("second simulated annealing\n");
    m_tCurrent = m_tInitial;
    unsigned int loopCounter = 0;
    while (m_tCurrent > m_tFinal)
        {
        if (loopCounter % 10 == 0)
            {
            printf("outer loop is %d\n", loopCounter);
            }
        // perform the simulated annealing
        bool keepAnnealing = true;
        unsigned int innerLoopCounter = 0;
        while (keepAnnealing)
            {
            // this is hitting an infinite loop with achievable probability...
            // set a loop max to just kick out?
            if (innerLoopCounter % 10 == 0)
                {
                printf("inner loop is %d\n", innerLoopCounter);
                }
            // generate trial rotation of global orientation
            // pick axis at random
            theta = 2.0 * M_PI * m_rngGen();
            phi = acos(2.0 * m_rngGen() - 1);
            axis.x = cosf(theta) * sinf(phi);
            axis.y = sinf(theta) * sinf(phi);
            axis.z = cos(phi);
            axis /= sqrt(dot(axis, axis));
            // pick an angle to rotate by, scaling by temperature
            // arbitrary scaling factor, can be changed, should be user-set
            // assume that the angle is very close
            angle = 0.2 * M_PI * m_rngGen();
            quat<float> newTrial = quat<float>::fromAxisAngle(axis, angle) * m_trial;
            l_p4SumNew0 = 0.0;
            // calculate the sum for the new trial vector
            parallel_for(blocked_range<size_t>(0,Np),
                         ComputeCubaticOrderParameter(l_p4SumNew0,
                                                      newTrial,
                                                      orientations));
            // normalize the sum
            l_p4SumNew0 = l_p4SumNew0 * (m_norm / Np);
            quat<float> newTrial1 = newTrial * q01;
            l_p4SumNew1 = 0.0;
            // calculate the sum for the new trial vector
            parallel_for(blocked_range<size_t>(0,Np),
                         ComputeCubaticOrderParameter(l_p4SumNew1,
                                                      newTrial1,
                                                      orientations));
            // normalize the sum
            l_p4SumNew1 = l_p4SumNew1 * (m_norm / Np);
            quat<float> newTrial2 = newTrial * q02;
            l_p4SumNew2 = 0.0;
            // calculate the sum for the new trial vector
            parallel_for(blocked_range<size_t>(0,Np),
                         ComputeCubaticOrderParameter(l_p4SumNew2,
                                                      newTrial2,
                                                      orientations));
            // normalize the sum
            l_p4SumNew2 = l_p4SumNew2 * (m_norm / Np);

            // calc the rms and optimize
            float l_RMS = sqrt((l_p4SumNew0*l_p4SumNew0 + l_p4SumNew1*l_p4SumNew1 + l_p4SumNew2*l_p4SumNew2)/3.0);
            // boltzmann criterion
            if (l_RMS >= m_RMS)
                {
                m_trial = newTrial;
                m_RMS = l_RMS;
                keepAnnealing = false;
                }
            else
                {
                float factor = exp(-(m_RMS - l_RMS)/m_tCurrent);
                float test = m_rngGen();
                if (factor >= test)
                    {
                    m_trial = newTrial;
                    m_RMS = l_RMS;
                    keepAnnealing = false;
                    }
                }
            innerLoopCounter++;
            }
        loopCounter++;
        // decrease temperature
        m_tCurrent *= m_scale;
        }
    m_cq0 = m_trial;
    m_cq1 = m_cq0 * q01;
    m_cq2 = m_cq0 * q02;

    m_p4Sum0 = l_p4SumNew0;
    m_p4Sum1 = l_p4SumNew1;
    m_p4Sum2 = l_p4SumNew2;

    m_p4_array[0] = m_p4Sum0;
    m_p4_array[1] = m_p4Sum1;
    m_p4_array[2] = m_p4Sum2;
    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
