#include "CubaticOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdlib.h>
#include <math.h>
#include <stdexcept>
#include <complex>
#include <tbb/tbb.h>

#include "HOOMDMath.h"
#include "VectorMath.h"

using namespace std;
using namespace tbb;

/*! \file CubaticOrderParameter.cc
    \brief Compute the global cubatic order parameter
*/

namespace freud { namespace order {

CubaticOrderParameter::CubaticOrderParameter(float tInitial, float tFinal, float scale, float norm)
    : m_box(trajectory::Box()), m_tInitial(tInitial), m_tFinal(tFinal), m_scale(scale), m_norm(norm), m_Np(0)
    {
    m_p4Sum = 0.0;
    m_p4SumNew = 0.0;
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
                float angle = acos(2.0 * powf(ref.s*trial.s + ref.v.x*trial.v.x + ref.v.y*trial.v.y + ref.v.z*trial.v.z, 2.0) - 1.0);
                float p4_i = 35.0*powf(cosf(angle), 4) - 30.0*powf(cosf(angle), 2) + 3.0;
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
    m_tCurrent = m_tInitial;

    // generate first trial rotation and normalize
    // http://mathworld.wolfram.com/SpherePointPicking.html
    float theta(2.0 * M_PI * rand());
    float phi(acos(2.0 * rand() - 1));
    vec3<float> axis(cosf(theta) * sinf(phi), sinf(theta) * sinf(phi), cos(phi));
    axis /= sqrt(dot(axis, axis));
    float angle(2.0 * M_PI * rand());
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

    while (m_tCurrent > m_tFinal)
        {
        // perform the simulated annealing
        bool keepAnnealing = true;
        while (keepAnnealing)
            {
            // generate new trial vector
            theta = 2.0 * M_PI * rand();
            phi = acos(2.0 * rand() - 1);
            axis.x = cosf(theta) * sinf(phi);
            axis.y = sinf(theta) * sinf(phi);
            axis.z = cos(phi);
            axis /= sqrt(dot(axis, axis));
            angle = 2.0 * M_PI * rand();
            quat<float> newTrial = quat<float>::fromAxisAngle(axis, angle);
            m_p4SumNew = 0.0;
            // calculate the sum for the new trial vector
            parallel_for(blocked_range<size_t>(0,Np),
                         ComputeCubaticOrderParameter(m_p4SumNew,
                                                      newTrial,
                                                      orientations));
            // normalize the sum
            m_p4SumNew = m_p4SumNew * (m_norm / Np);
            // perform checks
            if (m_p4SumNew >= m_p4Sum)
                {
                m_trial = newTrial;
                m_p4Sum = m_p4SumNew;
                keepAnnealing = false;
                }
            else
                {
                float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
                float test = rand();
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

    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
