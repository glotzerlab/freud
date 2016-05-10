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
using hoomd::matrix::quaternionFromExyz;

/*! \file LocalDescriptors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

LocalDescriptors::LocalDescriptors(const trajectory::Box& box,
        unsigned int nNeigh, unsigned int lmax, float rmax, bool negative_m):
    m_box(box), m_nNeigh(nNeigh), m_lmax(lmax), m_rmax(rmax),
    m_negative_m(negative_m), m_lc(box, rmax), m_Np(0), m_deficits()
    {
    m_deficits = 0;
    }

//! Utility function to sort a pair<float, pair<vec3<float>, unsigned int> > on the first
//! element of the pair
bool compareRsqVec(const pair<float, pair<vec3<float>, unsigned int> > &left,
    const pair<float, pair<vec3<float>, unsigned int> > &right)
    {
    return left.first < right.first;
    }

class ComputeLocalDescriptors
    {
private:
    const trajectory::Box& m_box;
    const unsigned int m_nNeigh;
    const unsigned int m_lmax;
    const unsigned int m_sphwidth;
    const float m_rmax;
    const bool m_negative_m;
    const locality::LinkCell& m_lc;
    const vec3<float> *m_r;
    const quat<float> *m_q;
    float *m_magrArray;
    quat<float> *m_qijArray;
    complex<float> *m_sphArray;
    tbb::atomic<unsigned int> &m_deficits;
public:
    ComputeLocalDescriptors(float *magrArray, quat<float> *qijArray,
        complex<float> *sphArray,
        tbb::atomic<unsigned int> &deficits,
        const trajectory::Box& box,
        const unsigned int nNeigh,
        const unsigned int lmax,
        const unsigned int sphwidth,
        const float rmax,
        const bool negative_m,
        const locality::LinkCell& lc,
        const vec3<float> *r, const quat<float> *q):
        m_box(box), m_nNeigh(nNeigh), m_lmax(lmax), m_sphwidth(sphwidth),
        m_rmax(rmax), m_negative_m(negative_m), m_lc(lc),
        m_r(r), m_q(q), m_magrArray(magrArray), m_qijArray(qijArray),
        m_sphArray(sphArray), m_deficits(deficits)
        {
        }

    void operator()( const blocked_range<size_t>& r ) const
        {
        float rmaxsq = m_rmax * m_rmax;
        // tuple<> is c++11, so for now just make a pair with pairs inside
        vector<pair<float, pair<vec3<float>, unsigned int> > > neighbors;

        fsph::PointSPHEvaluator<float> sph_eval(m_lmax);

        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            if(m_deficits > 0) break;
            neighbors.clear();

            //get cell point is in
            const vec3<float> ri(m_r[i]);
            // unsigned int ref_cell = m_lc.getCell(make_float3(ri.x, ri.y, ri.z));
            unsigned int ref_cell = m_lc.getCell(ri);
            unsigned int num_adjacent = 0;

            //loop over neighboring cells
            const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
            for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                {
                unsigned int neigh_cell = neigh_cells[neigh_idx];

                //iterate over particles in cell
                locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
                for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                    {
                    const vec3<float> rj(m_r[j]);
                    vec3<float> rij(rj - ri);

                    //compute r between the two particles
                    // const float3 wrapped(m_box.wrap(make_float3(rij.x, rij.y, rij.z)));
                    rij = m_box.wrap(rij);
                    // rij = vec3<float>(wrapped.x, wrapped.y, wrapped.z);
                    const float rsq(dot(rij, rij));

                    if (rsq < rmaxsq && rsq > 1e-6)
                        {
                        neighbors.push_back(pair<float, pair<vec3<float>, unsigned int> >(
                                rsq, pair<vec3<float>, unsigned int>(rij, j)));
                        num_adjacent++;
                        }
                    }
                }

            // Add to the deficit count if necessary
            if(num_adjacent < m_nNeigh)
                m_deficits += (m_nNeigh - num_adjacent);
            else
                {
                sort(neighbors.begin(), neighbors.end(), compareRsqVec);

                float inertiaTensor[3][3];
                for(size_t ii(0); ii < 3; ++ii)
                    for(size_t jj(0); jj < 3; ++jj)
                        inertiaTensor[ii][jj] = 0;

                for(size_t k(0); k < m_nNeigh; ++k)
                    {
                    const float rsq(neighbors[k].first);
                    const vec3<float> rvec(neighbors[k].second.first);

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
                    const vec3<float> rij(neighbors[k].second.first);
                    const vec3<float> bond(dot(eigenvec0, rij),
                                           dot(eigenvec1, rij),
                                           dot(eigenvec2, rij));

                    const float magR(sqrt(neighbors[k].first));
                    const float theta(atan2(bond.y, bond.x)); // theta in [0..2*pi]
                    const float phi(acos(bond.z/magR)); // phi in [0..pi]

                    sph_eval.compute(phi, theta);

                    std::copy(sph_eval.begin(m_negative_m), sph_eval.end(), &m_sphArray[sphCount]);
                    sphCount += m_sphwidth;
                    }
                }
            }
        }
    };

void LocalDescriptors::compute(const vec3<float> *r, const quat<float> *q, unsigned int Np)
    {
    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_magrArray = boost::shared_array<float>(new float[m_nNeigh*Np]);
        m_qijArray = boost::shared_array<quat<float> >(new quat<float>[m_nNeigh*Np]);
        m_sphArray = boost::shared_array<complex<float> >(new complex<float>[m_nNeigh*Np*getSphWidth()]);
        }

    // compute the order parameter
    do
        {
        // compute the cell list
        // m_lc.computeCellList((float3*)r, Np);
            m_lc.computeCellList(m_box, r, Np);

        m_deficits = 0;
        parallel_for(blocked_range<size_t>(0,Np),
            ComputeLocalDescriptors(m_magrArray.get(), m_qijArray.get(),
                m_sphArray.get(), m_deficits, m_box, m_nNeigh,
                m_lmax, getSphWidth(), m_rmax, m_negative_m, m_lc, r, q));

        // Increase m_rmax
        if(m_deficits > 0)
            {
            m_rmax *= 1.1;
            m_lc = locality::LinkCell(m_box, m_rmax);
            }
        } while(m_deficits > 0);

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
//     class_<LocalDescriptors>("LocalDescriptors", init<trajectory::Box&, unsigned int, unsigned int, float>())
//         .def("getBox", &LocalDescriptors::getBox, return_internal_reference<>())
//         .def("getNNeigh", &LocalDescriptors::getNNeigh)
//         .def("getLMax", &LocalDescriptors::getLMax)
//         .def("getRMax", &LocalDescriptors::getRMax)
//         .def("compute", &LocalDescriptors::computePy)
//         .def("getMagR", &LocalDescriptors::getMagRPy)
//         .def("getQij", &LocalDescriptors::getQijPy)
//         .def("getSph", &LocalDescriptors::getSphPy)
//         ;
//     }

}; }; // end namespace freud::order
