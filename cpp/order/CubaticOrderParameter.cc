#include "CubaticOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdlib.h>
#include <math.h>
#include <stdexcept>
#include <complex>
#include <tbb/tbb.h>

using namespace std;
using namespace tbb;

/*! \file CubaticOrderParameter.cc
    \brief Compute the global cubatic order parameter
*/

namespace freud { namespace order {

CubaticOrderParameter::CubaticOrderParameter(float tInitial, float tFinal, float scale, float norm)
    : m_box(trajectory::Box()), m_tInitial(tInitial), m_tFinal(tFinal), m_scale(scale), m_norm(norm), m_Np(0),
      m_p4Sum(0.0), m_p4SumNew(0.0)
    {
    }

CubaticOrderParameter::~CubaticOrderParameter()
    {
    delete m_lc;
    }

class ComputeCubaticOrderParameter
    {
    private:
        atomic<float> &m_p4Sum;
        const vec3<float> *m_trial;
        const vec3<float> *m_points;
    public:
        ComputeCubaticOrderParameter(atomic<float> &p4Sum,
                                 const vec3<float> *trial,
                                 const vec3<float> *points)
            : m_p4Sum(p4Sum), m_trial(trial), m_points(points)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float l_sum = 0.0;
            vec3<float> trial = m_trial[0];

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                vec3<float> ref = m_points[i];
                vec3<float> cross_i = cross(trial, ref);
                float angle = asin(sqrt(dot(cross_i, cross_i))/(sqrt(dot(trial))*sqrt(dot(ref))));
                float p4_i = 35.0*powf(cosf(angle), 4) - 30.0*powf(cosf(angle), 2) + 3.0;
                l_sum += p4_i;
                }
            m_p4Sum += l_sum
            }
    };

void CubaticOrderParameter::compute(trajectory::Box& box, const vec3<float> *points, unsigned int Np)
    {

    m_box = box;
    m_tCurrent = m_tInitial;

    // generate first trial and normalize
    vec3<float> trial(rand(), rand(), rand());
    trial /= sqrt(dot(trial, trial));

    while (m_tCurrent > m_tFinal)
        {
        // initialize the sums to zero
        m_p4Sum = 0.0;
        // calculate the sum for the current trial vector
        parallel_for(blocked_range<size_t>(0,Np),
                     ComputeCubaticOrderParameter(m_p4Sum,
                                                  trial,
                                                  points));
        // normalize the sum
        m_p4Sum *= (m_norm / Np);
        // perform the simulated annealing
        bool keepAnnealing = true;
        while (keepAnnealing)
            {
            // generate new trial vector
            vec3<float> newTrial(rand(), rand(), rand());
            newTrial /= sqrt(dot(newTrial, newTrial));
            m_p4SumNew = 0.0;
            // calculate the sum for the new trial vector
            parallel_for(blocked_range<size_t>(0,Np),
                         ComputeCubaticOrderParameter(m_p4SumNew,
                                                      newTrial,
                                                      points));
            // normalize the sum
            m_p4SumNew *= (m_norm / Np);
            // perform checks
            if (m_p4SumNew >= m_p4Sum)
                {
                trial = newTrial;
                keepAnnealing = false;
                }
            else
                {
                float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
                float test = rand();
                if (factor >= test)
                    {
                    trial = newTrial;
                    keepAnnealing = false;
                    }
                }
            }
        // decrease temperature
        m_tCurrent *= m_scale;
        // I think this won't calc the final value of the p4 sum
        // we should be able to make this faster by bypassing the trial parallel_for
        // since every time it changes we already know the values...
        }

    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
