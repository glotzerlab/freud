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

RotationalAutocorrelationFunction::RotationalAutocorrelationFunction(float rmax, float k)
    : m_Np(0)
    {
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






    // compute the cell list
    m_box = box;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_dr_array = std::shared_ptr<complex<float> >(new complex<float> [Np], std::default_delete<complex<float>[]>());
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Np),
        [=] (const blocked_range<size_t>& r)
        {
        size_t bond(nlist->find_first_index(r.begin()));
        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            m_dr_array.get()[i] = 0;
            vec3<float> ref = points[i];

            for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                {
                const size_t j(neighbor_list[2*bond + 1]);

                //compute r between the two particles
                vec3<float> delta = m_box.wrap(points[j] - ref);

                float rsq = dot(delta, delta);
                if (rsq > 1e-6)
                    {
                    //compute dr for neighboring particle(only constructed for 2d)
                    m_dr_array.get()[i] += complex<float>(delta.x, delta.y);
                    }
                }
            m_dr_array.get()[i] /= complex<float>(m_k);
            }
        });

    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
