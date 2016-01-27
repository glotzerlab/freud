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
    // create the random number generator
    // boost::mt19937 m_rng((unsigned int)std::time(0));
    // boost::uniform_real<float> m_rngDist(0,1);
    // boost::variate_generator< boost::mt19937&, boost::uniform_real<float> > m_rngGen(m_rng, m_rngDist);
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
    // check that this worked
    // vec3<float> v01 = rotate(q01, v0);
    // vec3<float> v02 = rotate(q02, v0);
    // printf("v1 = %f %f %f\n", v1.x, v1.y, v1.z);
    // printf("v01 = %f %f %f\n", v01.x, v01.y, v01.z);
    // printf("v2 = %f %f %f\n", v2.x, v2.y, v2.z);
    // printf("v02 = %f %f %f\n", v02.x, v02.y, v02.z);

    // find q0. this is found "at random"
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
    while (m_tCurrent > m_tFinal)
        {
        // perform the simulated annealing
        bool keepAnnealing = true;
        while (keepAnnealing)
            {
            // generatre trial rotation of global orientation
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
    // calc the sums, and return; will add refinement after (if this works)
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

    float m_RMS = sqrt((m_p4Sum0*m_p4Sum0 + m_p4Sum1*m_p4Sum1 + m_p4Sum2*m_p4Sum2)/3.0);
    // now we refine
    // initialize local sums to 0
    float l_p4Sum0 = 0.0;
    float l_p4Sum1 = 0.0;
    float l_p4Sum2 = 0.0;
    tbb::atomic<float> l_p4SumNew0;l_p4SumNew0 = 0.0;
    tbb::atomic<float> l_p4SumNew1;l_p4SumNew1 = 0.0;
    tbb::atomic<float> l_p4SumNew2;l_p4SumNew2 = 0.0;
    m_tCurrent = m_tInitial;
    m_trial = m_cq0;
    while (m_tCurrent > m_tFinal)
        {
        // perform the simulated annealing
        bool keepAnnealing = true;
        while (keepAnnealing)
            {
                        // generatre trial rotation of global orientation
            // pick axis at random
            theta = 2.0 * M_PI * m_rngGen();
            phi = acos(2.0 * m_rngGen() - 1);
            axis.x = cosf(theta) * sinf(phi);
            axis.y = sinf(theta) * sinf(phi);
            axis.z = cos(phi);
            axis /= sqrt(dot(axis, axis));
            // pick an angle to rotate by, scaling by temperature
            // arbitrary scaling factor, can be changed, should be user-set
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
            }
        // decrease temperature
        m_tCurrent *= m_scale;
        }
    m_cq0 = m_trial;
    m_cq1 = m_cq0 * q01;
    m_cq2 = m_cq0 * q02;

    m_p4Sum0 = l_p4SumNew0;
    m_p4Sum1 = l_p4SumNew1;
    m_p4Sum2 = l_p4SumNew2;
    // now we have one orientation, we can generate the other two from our knowledge of the
    // cubatic symmetry
    // m_cq1 = m_cq0 * q01;
    // m_cq2 = m_cq0 * q02;

    // m_tCurrent = m_tInitial;
    // angle = 2.0 * M_PI * m_rngGen();
    // // generate the quaternion
    // m_trial = quat<float>::fromAxisAngle(axis, angle) * m_cq0;
    // // initialize the sum to zero
    // m_p4Sum = 0.0;
    // // calculate the sum for the current trial vector
    // parallel_for(blocked_range<size_t>(0,Np),
    //              ComputeCubaticOrderParameter(m_p4Sum,
    //                                           m_trial,
    //                                           orientations));
    // // normalize the sum
    // m_p4Sum = m_p4Sum * (m_norm / Np);
    //
    // while (m_tCurrent > m_tFinal)
    //     {
    //     // perform the simulated annealing
    //     bool keepAnnealing = true;
    //     while (keepAnnealing)
    //         {
    //         // generatre trial rotation of global orientation
    //         angle = m_tCurrent * M_PI * m_rngGen();
    //         quat<float> newTrial = quat<float>::fromAxisAngle(axis, angle) * m_trial;
    //         m_p4SumNew = 0.0;
    //         // calculate the sum for the new trial vector
    //         parallel_for(blocked_range<size_t>(0,Np),
    //                      ComputeCubaticOrderParameter(m_p4SumNew,
    //                                                   newTrial,
    //                                                   orientations));
    //         // normalize the sum
    //         m_p4SumNew = m_p4SumNew * (m_norm / Np);
    //         // boltzmann criterion
    //         if (m_p4SumNew >= m_p4Sum)
    //             {
    //             m_trial = newTrial;
    //             m_p4Sum = m_p4SumNew;
    //             keepAnnealing = false;
    //             }
    //         else
    //             {
    //             float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
    //             float test = m_rngGen();
    //             if (factor >= test)
    //                 {
    //                 m_trial = newTrial;
    //                 m_p4Sum = m_p4SumNew;
    //                 keepAnnealing = false;
    //                 }
    //             }
    //         }
    //     // decrease temperature
    //     m_tCurrent *= m_scale;
    //     }
    //
    // m_cq1 = m_trial;
    // m_p4Sum1 = m_p4Sum;
    // // create the resultant orientation
    // m_cq2 = m_cq1*m_cq0;
    // // now we should be able to run the final quaternion through to get the val
    // m_p4Sum = 0.0;
    // parallel_for(blocked_range<size_t>(0,Np),
    //              ComputeCubaticOrderParameter(m_p4Sum,
    //                                           m_cq2,
    //                                           orientations));
    // m_p4Sum = m_p4Sum * (m_norm / Np);
    // m_p4Sum2 = m_p4Sum;

    // old stuff below, looks like it should work

    // printf("finding P4_0\n");
    // generate first trial rotation and normalize
    // http://mathworld.wolfram.com/SpherePointPicking.html
    // float theta(2.0 * M_PI * m_rngGen());
    // float phi(acos(2.0 * m_rngGen() - 1));
    // vec3<float> axis(cosf(theta) * sinf(phi), sinf(theta) * sinf(phi), cos(phi));
    // axis /= sqrt(dot(axis, axis));
    // float angle(2.0 * M_PI * m_rngGen());
    // // generate the quaternion
    // m_trial = quat<float>::fromAxisAngle(axis, angle);
    // // initialize the sum to zero
    // m_p4Sum = 0.0;
    // // calculate the sum for the current trial vector
    // parallel_for(blocked_range<size_t>(0,Np),
    //              ComputeCubaticOrderParameter(m_p4Sum,
    //                                           m_trial,
    //                                           orientations));
    // // normalize the sum
    // m_p4Sum = m_p4Sum * (m_norm / Np);
    //
    // m_tCurrent = m_tInitial;
    // while (m_tCurrent > m_tFinal)
    //     {
    //     // perform the simulated annealing
    //     bool keepAnnealing = true;
    //     while (keepAnnealing)
    //         {
    //         // generatre trial rotation of global orientation
    //         // pick axis at random
    //         theta = 2.0 * M_PI * m_rngGen();
    //         phi = acos(2.0 * m_rngGen() - 1);
    //         axis.x = cosf(theta) * sinf(phi);
    //         axis.y = sinf(theta) * sinf(phi);
    //         axis.z = cos(phi);
    //         axis /= sqrt(dot(axis, axis));
    //         // pick an angle to rotate by, scaling by temperature
    //         // arbitrary scaling factor, can be changed, should be user-set
    //         angle = 10.0 * m_tCurrent * M_PI * m_rngGen();
    //         quat<float> newTrial = quat<float>::fromAxisAngle(axis, angle) * m_trial;
    //         m_p4SumNew = 0.0;
    //         // calculate the sum for the new trial vector
    //         parallel_for(blocked_range<size_t>(0,Np),
    //                      ComputeCubaticOrderParameter(m_p4SumNew,
    //                                                   newTrial,
    //                                                   orientations));
    //         // normalize the sum
    //         m_p4SumNew = m_p4SumNew * (m_norm / Np);
    //         // boltzmann criterion
    //         if (m_p4SumNew >= m_p4Sum)
    //             {
    //             m_trial = newTrial;
    //             m_p4Sum = m_p4SumNew;
    //             keepAnnealing = false;
    //             }
    //         else
    //             {
    //             float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
    //             float test = m_rngGen();
    //             if (factor >= test)
    //                 {
    //                 m_trial = newTrial;
    //                 m_p4Sum = m_p4SumNew;
    //                 keepAnnealing = false;
    //                 }
    //             }
    //         }
    //     // decrease temperature
    //     m_tCurrent *= m_scale;
    //     }
    // m_cq0 = m_trial;
    // m_p4Sum0 = m_p4Sum;
    // // now, do the same for the other two orientations
    // m_tCurrent = m_tInitial;
    //
    // // this should be a plane equation
    // // printf("finding P4_1\n");
    // bool keepGenerating = true;
    // while (keepGenerating)
    //     {
    //     // printf("generating a new vector\n");
    //     // generate first trial rotation and normalize
    //     // http://mathworld.wolfram.com/SpherePointPicking.html
    //     theta = 2.0 * M_PI * m_rngGen();
    //     phi = acos(2.0 * m_rngGen() - 1);
    //     axis.x = cosf(theta) * sinf(phi);
    //     axis.y = sinf(theta) * sinf(phi);
    //     axis.z = cosf(phi);
    //     axis /= sqrt(dot(axis, axis));
    //     angle = 2.0 * M_PI * m_rngGen();
    //     // generate the quaternion
    //     m_trial = quat<float>::fromAxisAngle(axis, angle);
    //     // check that it is close pi/2 away
    //     // this is probably going to have issues with signs
    //     float angleCheck = acos(2.0 * powf(m_cq0.s*m_trial.s + m_cq0.v.x*m_trial.v.x + m_cq0.v.y*m_trial.v.y + m_cq0.v.z*m_trial.v.z, 2.0) - 1.0);
    //     // 0.2 is arbitrary as well
    //     if (abs((M_PI/2.0) - angleCheck) < 0.2)
    //         {
    //         keepGenerating = false;
    //         }
    //     }
    // // initialize the sum to zero
    // m_p4Sum = 0.0;
    // // calculate the sum for the current trial vector
    // parallel_for(blocked_range<size_t>(0,Np),
    //              ComputeCubaticOrderParameter(m_p4Sum,
    //                                           m_trial,
    //                                           orientations));
    // // normalize the sum
    // m_p4Sum = m_p4Sum * (m_norm / Np);
    //
    // while (m_tCurrent > m_tFinal)
    //     {
    //     // perform the simulated annealing
    //     bool keepAnnealing = true;
    //     while (keepAnnealing)
    //         {
    //         // generatre trial rotation of global orientation
    //         keepGenerating = true;
    //         quat<float> newTrial;
    //         while (keepGenerating)
    //             {
    //             // printf("generating a new trial vector\n");
    //             // pick axis at random
    //             theta = 2.0 * M_PI * m_rngGen();
    //             phi = acos(2.0 * m_rngGen() - 1);
    //             axis.x = cosf(theta) * sinf(phi);
    //             axis.y = sinf(theta) * sinf(phi);
    //             axis.z = cos(phi);
    //             axis /= sqrt(dot(axis, axis));
    //             // pick an angle to rotate by, scaling by temperature
    //             angle = m_tCurrent * M_PI * m_rngGen();
    //             // check that it is close pi/2 away
    //             newTrial = quat<float>::fromAxisAngle(axis, angle) * m_trial;
    //             float angleCheck = acos(2.0 * powf(m_cq0.s*newTrial.s + m_cq0.v.x*newTrial.v.x + m_cq0.v.y*newTrial.v.y + m_cq0.v.z*newTrial.v.z, 2.0) - 1.0);
    //             // printf("angle generated = %f, angleCheck = %f\n", angle, angleCheck);
    //             // I don't think this angle check is good, need to figure out something else
    //             if (abs((M_PI/2.0) - angleCheck) < 0.2)
    //                 {
    //                 keepGenerating = false;
    //                 }
    //             }
    //         m_p4SumNew = 0.0;
    //         // calculate the sum for the new trial vector
    //         parallel_for(blocked_range<size_t>(0,Np),
    //                      ComputeCubaticOrderParameter(m_p4SumNew,
    //                                                   newTrial,
    //                                                   orientations));
    //         // normalize the sum
    //         m_p4SumNew = m_p4SumNew * (m_norm / Np);
    //         // boltzmann criterion
    //         if (m_p4SumNew >= m_p4Sum)
    //             {
    //             m_trial = newTrial;
    //             m_p4Sum = m_p4SumNew;
    //             keepAnnealing = false;
    //             }
    //         else
    //             {
    //             float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
    //             float test = m_rngGen();
    //             if (factor >= test)
    //                 {
    //                 m_trial = newTrial;
    //                 m_p4Sum = m_p4SumNew;
    //                 keepAnnealing = false;
    //                 }
    //             }
    //         }
    //     // decrease temperature
    //     m_tCurrent *= m_scale;
    //     }
    //
    // m_cq1 = m_trial;
    // m_p4Sum1 = m_p4Sum;
    //
    // // check that cq0 and cq1 are pi/2 off
    // // printf("finding P4_2\n");
    // // now, do the same for the final orientation
    // m_tCurrent = m_tInitial;
    // // determine final quaternion guess
    // // this is not correct...as far as I can tell...ugh
    // float angle0 = 2.0 * acos(m_cq0.s);
    // vec3<float> axis0(m_cq0.v.x / sin(angle0/2.0), m_cq0.v.y / sin(angle0/2.0), m_cq0.v.z / sin(angle0/2.0));
    // float angle1 = 2.0 * acos(m_cq1.s);
    // vec3<float> axis1(m_cq1.v.x / sin(angle1/2.0), m_cq1.v.y / sin(angle1/2.0), m_cq1.v.z / sin(angle1/2.0));
    // // find the final axis
    // vec3<float> axis2 = cross(axis0, axis1);
    // // fixed the axis -> axis 2...may have been part of the issue?
    // axis2 /= sqrt(dot(axis2, axis2));
    // // looks like maybe this is 2x too big?
    // // float angle2 = 2.0 * atan(-1.0 / (tan(angle0/2.0) * dot(axis0, axis2)));
    // float angle2 = atan(-1.0 / (tan(angle0/2.0) * dot(axis0, axis2)));
    // // printf("angle2 = %f\n", angle2);
    // // check that the new quaternion is pi/2 away
    // m_trial = quat<float>::fromAxisAngle(axis2, angle2);
    // // do it the new way
    // m_trial = m_cq1 * m_cq0;
    // angle = acos(2.0 * powf(m_cq0.s*m_trial.s + m_cq0.v.x*m_trial.v.x + m_cq0.v.y*m_trial.v.y + m_cq0.v.z*m_trial.v.z, 2.0) - 1.0);
    // // printf("angle = %f\n", angle);
    // // // this check may not be correct
    // // if (abs(angle - (M_PI/2.0) > 0.2))
    // //     {
    // //     printf("angle is bad\n");
    // //     }
    // // angle = acos(2.0 * powf(m_cq1.s*m_trial.s + m_cq1.v.x*m_trial.v.x + m_cq1.v.y*m_trial.v.y + m_cq1.v.z*m_trial.v.z, 2.0) - 1.0);
    // // printf("angle = %f\n", angle);
    // // if (abs(angle - (M_PI/2.0) > 0.2))
    // //     {
    // //     printf("angle is bad\n");
    // //     }
    // // return;
    //
    // // initialize the sum to zero
    // m_p4Sum = 0.0;
    // // calculate the sum for the current trial vector
    // parallel_for(blocked_range<size_t>(0,Np),
    //              ComputeCubaticOrderParameter(m_p4Sum,
    //                                           m_trial,
    //                                           orientations));
    // // normalize the sum
    // m_p4Sum = m_p4Sum * (m_norm / Np);
    //
    // while (m_tCurrent > m_tFinal)
    //     {
    //     // perform the simulated annealing
    //     bool keepAnnealing = true;
    //     while (keepAnnealing)
    //         {
    //         // generatre trial rotation of global orientation
    //         quat<float> newTrial;
    //         keepGenerating = true;
    //         while (keepGenerating)
    //             {
    //             // pick axis at random
    //             theta = 2.0 * M_PI * m_rngGen();
    //             phi = acos(2.0 * m_rngGen() - 1);
    //             axis.x = cosf(theta) * sinf(phi);
    //             axis.y = sinf(theta) * sinf(phi);
    //             axis.z = cos(phi);
    //             axis /= sqrt(dot(axis, axis));
    //             // pick an angle to rotate by, scaling by temperature
    //             angle = 10.0 * m_tCurrent * M_PI * m_rngGen();
    //             // check that it is close pi/2 away
    //             newTrial = quat<float>::fromAxisAngle(axis, angle) * m_trial;
    //             float angleCheck0 = acos(2.0 * powf(m_cq0.s*newTrial.s + m_cq0.v.x*newTrial.v.x + m_cq0.v.y*newTrial.v.y + m_cq0.v.z*newTrial.v.z, 2.0) - 1.0);
    //             float angleCheck1 = acos(2.0 * powf(m_cq1.s*newTrial.s + m_cq1.v.x*newTrial.v.x + m_cq1.v.y*newTrial.v.y + m_cq1.v.z*newTrial.v.z, 2.0) - 1.0);
    //             if ((angleCheck0 > ((M_PI/2.0) - 0.2)) || (angleCheck1 > ((M_PI/2.0) - 0.2)))
    //                 {
    //                 keepGenerating = false;
    //                 }
    //             }
    //         m_p4SumNew = 0.0;
    //         // calculate the sum for the new trial vector
    //         parallel_for(blocked_range<size_t>(0,Np),
    //                      ComputeCubaticOrderParameter(m_p4SumNew,
    //                                                   newTrial,
    //                                                   orientations));
    //         // normalize the sum
    //         m_p4SumNew = m_p4SumNew * (m_norm / Np);
    //         // boltzmann criterion
    //         if (m_p4SumNew >= m_p4Sum)
    //             {
    //             m_trial = newTrial;
    //             m_p4Sum = m_p4SumNew;
    //             keepAnnealing = false;
    //             }
    //         else
    //             {
    //             float factor = exp(-(m_p4Sum - m_p4SumNew)/m_tCurrent);
    //             float test = m_rngGen();
    //             if (factor >= test)
    //                 {
    //                 m_trial = newTrial;
    //                 m_p4Sum = m_p4SumNew;
    //                 keepAnnealing = false;
    //                 }
    //             }
    //         }
    //     // decrease temperature
    //     m_tCurrent *= m_scale;
    //     }
    //
    // m_cq2 = m_trial;
    // m_p4Sum2 = m_p4Sum;
    m_p4_array[0] = m_p4Sum0;
    m_p4_array[1] = m_p4Sum1;
    m_p4_array[2] = m_p4Sum2;
    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
