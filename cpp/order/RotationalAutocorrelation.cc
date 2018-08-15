// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#include <math.h>

#include "RotationalAutocorrelationFunction.h"

using namespace std;
using namespace tbb;

/*! \file RotationalAutocorrelationFunction.h
    \brief Compute the rotational autocorrelation function for a system
    against a reference set of orientations
*/

namespace freud { namespace order {

RotationalAutocorrelationFunction::RotationalAutocorrelationFunction(float l)
    : m_Np(0)
    {
      //TKTK: probably need to delete the below two lines
      //m_RAi = std::shared_ptr<complex<float> >(new complex<float> [m_Np]);
      //memset((void*)m_RAi.get(), 0, sizeof(complex(float))*m_Np)
    }

RotationalAutocorrelationFunction::~RotationalAutocorrelationFunction()
    {
    }

//Going to define some functions that are used in multiple spots

std::pair<std::complex<float>, std::complex<float> > quat_to_greek(const quat<float> &q)
  {
    const std::complex<float> xi(q.v.x + q.v.y*std::complex<float>(0, 1));
    const std::complex<float> zeta(q.v.z + q.s*std::complex<float>(0, 1));

    return std::pair<std::complex<float>, std::complex<float> >(xi, zeta);
  }

//Defining a function to handle complex numbers being raised to some power.
//Doing this to get around a specific case where pow((0,0), 0) returns (nan, nan)

std::complex<float> cpow(std::complex<float> base, float p)
  {
    if (p == 0)
    {
        return std::complex<float> (1,0);
    }
    else
    {
        return pow(base, p);
  }

//Defining the factorial function to rely on std's gamma functions
float factorial(int n)
  {
    return tgamma(n+1);
  }

//Function to calculate the hypersphere harmonic
std::complex<float> hypersphere_harmonic(const std::complex<float> xi, std::complex<float> zeta,
                                         const int l, const int m1, const int m2
                                         )
  {
    const int a = -(m1 - l/2);
    const int b = -(m2 - l/2);

    std::complex<float> sum_tracker(0,0);
    //Doing a summation over non-negative exponents
    for ( int k=0; k <= min(a, b); k++)
      {
        sum_tracker = sum_tracker + cpow(std::conj(xi),k) * cpow(zeta, b-k) *
                            cpow(std::conj(zeta), a-k), * cpow(-xi, l_k-a-b) /
                            factorial(k) / factorial(l+k-a-b) /
                            factorial(a-k) / factorial(b-k);
      }

      //Use cpow(expression, 1/2) as a way to compute the square root
    sum_tracker = sum_tracker * cpow(factorial(a) * factorial(l-a) *
                                     factorial(b) * factorial(l-b) / float(l+1),
                                   1/2);
    return sum_tracker;

  }

void RotationalAutocorrelationFunction::compute(
                const quat<float> *ref_ors,
                const quat<float> *ors,
                unsigned int Np)
    {

      //get the size of the orientation array for looping purposes
      m_Np = Np;

      //TKTK: make sure this is the best way to initialize an array of complex numbers
      //m_RAi = std::shared_ptr<complex<float> >(new complex<float> [m_Np]);

      //TKTK: not sure if this line is useful
      //memset((void*)m_RAi.get(), 0, sizeof(complex<float>)*m_Np);

      const size_t Np;
      // reallocate the output array if it is not the right size
      if (Np != m_Np)
          {
          m_RA_array = std::shared_ptr< std::complex<float> >(
                  new complex [Np],
                  std::default_delete<float[]>());
          }
      //Compute relevant values for all orientations in the system
      parallel_for(blocked_range<size_t>(0,Np),
          [=] (const blocked_range<size_t>& r)
          {

          assert(ref_ors);
          assert(ors);
          assert(Np > 0);
          assert(ref_ors.size == ors.size);

          for (size_t i=r.begin(); i!=r.end(); ++i)
              {
              quat<float> q_i(ref_ors[i]);
              quat<float> q_t(ors[i]);
              //Transform the orientation quaternions for normalization purposes
              quat<float> qq_0 = conj(q_i) * q_i;
              quat<float> qq_1 = conj(q_i) * q_t;

              std::pair<std::complex<float>, std::complex<float> > angle_0 =
                                                            quat_to_greek(qq_0);
              std::pair<std::complex<float>, std::complex<float> > angle_1 =
                                                            quat_to_greek(qq_1);
              //At this point, we've transformed the quaternions to xi and zeta


              signed int m1;
              signed int m2;
              //Need to loop through quantum numbers m1 and m2 which depend on l
              for (m1 = -1*m_l/2; m1<= m_l/2; m1++)
                {
                  for (m2 = -1*m_l/2; m1<= m_l/2; m2++)
                  {
                    std::complex <float> combined_value = std::conj(
                      hypersphere_harmonic(angle_0.first, angle_0.second,
                        m_l, m1, m2)
                       * hypersphere_harmonic(angle_1.first, angle_1.second,
                        m_1, m1, m2);
                    m_RA_array.get()[i] = combined_value;
                  }
                }
              }
          });



      // save the last computed number of particles
      m_Np = Np;
    };

std::shared_ptr<float> RotationalAutocorrelationFunction::getRotationalAutocorrelationFunction()
    {
    return m_Ft;
    }



}; }; // end namespace freud::order
