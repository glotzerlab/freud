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

class ComputeLocalDescriptors
    {
private:
    const box::Box& m_box;
    const unsigned int m_nNeigh;
    const unsigned int m_neighmax;
    const unsigned int m_lmax;
    const unsigned int m_sphwidth;
    const bool m_negative_m;
    const vec3<float> *m_r;
    const unsigned int *m_neighborList;
    const float *m_rsqArray;
    complex<float> *m_sphArray;
public:
    ComputeLocalDescriptors(
        complex<float> *sphArray,
        const box::Box& box,
        const unsigned int nNeigh,
        const unsigned int neighmax,
        const unsigned int lmax,
        const unsigned int sphwidth,
        const bool negative_m,
        const vec3<float> *r,
        const unsigned int *neighborList, const float *rsqArray):
        m_box(box), m_nNeigh(nNeigh), m_neighmax(neighmax), m_lmax(lmax), m_sphwidth(sphwidth),
        m_negative_m(negative_m),
        m_r(r), m_neighborList(neighborList), m_rsqArray(rsqArray),
        m_sphArray(sphArray)
        {
        }

    void operator()( const blocked_range<size_t>& r ) const
        {
        fsph::PointSPHEvaluator<float> sph_eval(m_lmax);
        Index2D idx_nlist(m_neighmax, 0);

        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            const vec3<float> r_i(m_r[i]);

            float inertiaTensor[3][3];
            for(size_t ii(0); ii < 3; ++ii)
                for(size_t jj(0); jj < 3; ++jj)
                    inertiaTensor[ii][jj] = 0;

            for(size_t k(0); k < m_nNeigh; ++k)
                {
                const float rsq(m_rsqArray[idx_nlist(k, i)]);
                const vec3<float> r_j(m_r[m_neighborList[idx_nlist(k, i)]]);
                const vec3<float> rvec(m_box.wrap(r_j - r_i));

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

            unsigned int sphCount(i*m_nNeigh*m_sphwidth);

            for(size_t k(0); k < m_nNeigh; ++k)
                {
                const float rsq(m_rsqArray[idx_nlist(k, i)]);
                const vec3<float> r_j(m_r[m_neighborList[idx_nlist(k, i)]]);
                const vec3<float> rij(m_box.wrap(r_j - r_i));
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
                sphCount += m_sphwidth;
                }
            }
        }
    };

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
        ComputeLocalDescriptors(
            m_sphArray.get(), box, nNeigh, m_neighmax,
            m_lmax, getSphWidth(), m_negative_m, r,
            m_nn.getNeighborList().get(), m_nn.getRsqList().get()));

    // save the last computed number of particles
    m_Np = Np;
    }

// void LocalDescriptors::computePy(boost::python::numeric::array r,
//     boost::python::numeric::array q)
//     {
//     //validate input type and rank
//     num_util::check_type(r, NPY_FLOAT);
//     num_util::check_rank(r, 2);
//     num_util::check_type(q, NPY_FLOAT);
//     num_util::check_rank(q, 2);

//     // validate that the 2nd dimension is only 3 for r and 4 for q
//     num_util::check_dim(r, 1, 3);
//     num_util::check_dim(q, 1, 4);
//     unsigned int Np = num_util::shape(r)[0];

//     if(num_util::shape(r)[0] != num_util::shape(q)[0])
//         throw runtime_error("Position and quaternion arrays must have the same length!");

//     // get the raw data pointers and compute order parameter
//     vec3<float>* r_raw = (vec3<float>*) num_util::data(r);
//     quat<float>* q_raw = (quat<float>*) num_util::data(q);

//     // compute the order parameter with the GIL released
//         {
//         util::ScopedGILRelease gil;
//         compute(r_raw, q_raw, Np);
//         }
//     }

// void export_LocalDescriptors()
//     {
//     class_<LocalDescriptors>("LocalDescriptors", init<box::Box&, unsigned int, unsigned int, float>())
//         .def("getBox", &LocalDescriptors::getBox, return_internal_reference<>())
//         .def("getNNeigh", &LocalDescriptors::getNNeigh)
//         .def("getLMax", &LocalDescriptors::getLMax)
//         .def("compute", &LocalDescriptors::computePy)
//         .def("getMagR", &LocalDescriptors::getMagRPy)
//         .def("getSph", &LocalDescriptors::getSphPy)
//         ;
//     }

}; }; // end namespace freud::order
