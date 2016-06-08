#include <algorithm>
#include <stdexcept>
#include <complex>
#include <utility>
#include <vector>
#include <tbb/tbb.h>

#include "LocalDescriptors.h"
#include "ScopedGILRelease.h"
#include "HOOMDMatrix.h"

using namespace std;
using namespace tbb;
using hoomd::matrix::diagonalize;

/*! \file LocalDescriptors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

LocalDescriptors::LocalDescriptors(
        unsigned int neighmax, unsigned int lmax, float rmax, bool negative_m):
    m_neighmax(neighmax), m_lmax(lmax),
    m_negative_m(negative_m), m_nn(rmax, neighmax), m_Np(0), m_nNeigh(0)
    {
    }


void LocalDescriptors::computeNList(const box::Box& box, const vec3<float> *r, unsigned int Np)
    {
    m_nn.compute(box, r, Np, r, Np);
    }

void LocalDescriptors::compute(const box::Box& box, unsigned int nNeigh, const vec3<float> *r, unsigned int Np)
    {
    if(m_nn.getNp() != Np)
        throw runtime_error("Must call computeNList() before compute");

    // reallocate the output array if it is not the right size
    if (Np != m_Np || nNeigh != m_nNeigh)
        {
        m_sphArray = boost::shared_array<complex<float> >(new complex<float>[nNeigh*Np*getSphWidth()]);
        m_nNeigh = nNeigh;
        }

    parallel_for(blocked_range<size_t>(0,Np),
        [=] (const blocked_range<size_t>& br)
        {
        fsph::PointSPHEvaluator<float> sph_eval(m_lmax);
        Index2D idx_nlist(m_neighmax, 0);

        for(size_t i=br.begin(); i!=br.end(); ++i)
            {
            const vec3<float> r_i(r[i]);

            float inertiaTensor[3][3];
            for(size_t ii(0); ii < 3; ++ii)
                for(size_t jj(0); jj < 3; ++jj)
                    inertiaTensor[ii][jj] = 0;

            for(size_t k(0); k < nNeigh; ++k)
                {
                const float rsq(m_nn.getRsqList().get()[idx_nlist(k, i)]);
                const vec3<float> r_j(r[m_nn.getNeighborList().get()[idx_nlist(k, i)]]);
                const vec3<float> rvec(box.wrap(r_j - r_i));

                for(size_t ii(0); ii < 3; ++ii)
                    inertiaTensor[ii][ii] += rsq;

                inertiaTensor[0][0] -= rvec.x*rvec.x;
                inertiaTensor[0][1] -= rvec.x*rvec.y;
                inertiaTensor[0][2] -= rvec.x*rvec.z;
                inertiaTensor[1][0] -= rvec.x*rvec.y;
                inertiaTensor[1][1] -= rvec.y*rvec.y;
                inertiaTensor[1][2] -= rvec.y*rvec.z;
                inertiaTensor[2][0] -= rvec.x*rvec.z;
                inertiaTensor[2][1] -= rvec.y*rvec.z;
                inertiaTensor[2][2] -= rvec.z*rvec.z;
                }

            float eigenvalues[3];
            float eigenvectors[3][3];

            diagonalize(inertiaTensor, eigenvalues, eigenvectors);

            // Sort eigenvalues and eigenvectors so that
            // eigenvalues is in ascending order. This is
            // a kind of gross way to do it, but it gets
            // the job done.
            if(eigenvalues[0] > eigenvalues[1])
                {
                std::swap(eigenvalues[0], eigenvalues[1]);
                for(size_t ii(0); ii < 3; ++ii)
                    std::swap(eigenvectors[0][ii], eigenvectors[1][ii]);
                }
            if(eigenvalues[1] > eigenvalues[2])
                {
                std::swap(eigenvalues[1], eigenvalues[2]);
                for(size_t ii(0); ii < 3; ++ii)
                    std::swap(eigenvectors[1][ii], eigenvectors[2][ii]);
                }
            if(eigenvalues[0] > eigenvalues[1])
                {
                std::swap(eigenvalues[0], eigenvalues[1]);
                for(size_t ii(0); ii < 3; ++ii)
                    std::swap(eigenvectors[0][ii], eigenvectors[1][ii]);
                }

            const vec3<float> eigenvec0(eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]);
            const vec3<float> eigenvec1(eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1]);
            const vec3<float> eigenvec2(eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2]);

            unsigned int sphCount(i*nNeigh*getSphWidth());

            for(size_t k(0); k < nNeigh; ++k)
                {
                const float rsq(m_nn.getRsqList().get()[idx_nlist(k, i)]);
                const vec3<float> r_j(r[m_nn.getNeighborList().get()[idx_nlist(k, i)]]);
                const vec3<float> rij(box.wrap(r_j - r_i));
                const vec3<float> bond(dot(eigenvec0, rij),
                                       dot(eigenvec1, rij),
                                       dot(eigenvec2, rij));

                const float magR(sqrt(rsq));
                float theta(atan2(bond.y, bond.x)); // theta in [-pi..pi] initially
                if(theta < 0)
                    theta += 2*M_PI; // move theta into [0..2*pi]
                const float phi(acos(bond.z/magR)); // phi in [0..pi]

                sph_eval.compute(phi, theta);

                std::copy(sph_eval.begin(m_negative_m), sph_eval.end(), &m_sphArray[sphCount]);
                sphCount += getSphWidth();
                }
            }
        });

    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
