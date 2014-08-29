#include <algorithm>
#include <stdexcept>
#include <complex>
#include <utility>
#include <vector>
#include <tbb/tbb.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "NearestNeighbors.h"
#include "ScopedGILRelease.h"
#include "HOOMDMatrix.h"

using namespace std;
using namespace boost::python;
using namespace tbb;
using hoomd::matrix::diagonalize;
using hoomd::matrix::quaternionFromExyz;

/*! \file NearestNeighbors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

NearestNeighbors::NearestNeighbors(const trajectory::Box& box, unsigned int nNeigh, float rmax):
    m_box(box), m_nNeigh(nNeigh), m_rmax(rmax), m_lc(box, rmax), m_deficits()
    {
    m_deficits = 0;
    }

//! Utility function to sort a pair<float, pair<vec3<float>, unsigned int> > on the first
//! element of the pair
bool compareRsqVectors(const pair<float, pair<vec3<float>, unsigned int> > &left,
                       const pair<float, pair<vec3<float>, unsigned int> > &right)
    {
    return left.first < right.first;
    }

class ComputeNearestNeighbors
    {
private:
    const trajectory::Box& m_box;
    const unsigned int m_nNeigh;
    const float m_rmax;
    const locality::LinkCell& m_lc;
    const vec3<float> *m_r;
    atomic<unsigned int> &m_deficits;
    atomic<unsigned int> *m_neighbor_array;
public:
    ComputeNearestNeighbors(atomic<unsigned int> &deficits,
                             atomic<unsigned int> *neighbor_array,
                             const trajectory::Box& box,
                             const unsigned int nNeigh,
                             const float rmax,
                             const locality::LinkCell& lc,
                             const vec3<float> *r):
        m_box(box), m_nNeigh(nNeigh), m_rmax(rmax), m_lc(lc),
        m_r(r), m_deficits(deficits), m_neighbor_array(neighbor_array)
        {
        }

    void operator()( const blocked_range<size_t>& r ) const
        {
        float rmaxsq = m_rmax * m_rmax;
        // tuple<> is c++11, so for now just make a pair with pairs inside
        // this data structure holds rsq, pos, idx
        vector<pair<float, pair<vec3<float>, unsigned int> > > neighbors;

        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            neighbors.clear();

            //get cell point is in
            const vec3<float> ri(m_r[i]);
            unsigned int ref_cell = m_lc.getCell(make_float3(ri.x, ri.y, ri.z));
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
                    const float3 wrapped(m_box.wrap(make_float3(rij.x, rij.y, rij.z)));
                    rij = vec3<float>(wrapped.x, wrapped.y, wrapped.z);
                    const float rsq(dot(rij, rij));

                    // adds all neighbors within rsq to list of possible neighbors
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
                // sort based on rsq
                sort(neighbors.begin(), neighbors.end(), compareRsqVectors);
                       for (unsigned int k = 0; k < m_nNeigh; k++)
                    {
                    // put the idx into the neighbor array
                    m_neighbor_array[i*m_nNeigh + k] = neighbors[k].second.second;
                    }
                }
            }
        }
    };

void NearestNeighbors::compute(const vec3<float> *r, unsigned int Np)
    {
    // find the nearest neighbors
    do
        {
        // compute the cell list
        m_lc.computeCellList((float3*)r, Np);

        // create and populate the idx array
        m_neighbor_array = boost::shared_array<unsigned int>(new unsigned int[Np * m_nNeigh]);
        memset((void*)m_neighbor_array.get(), 0, sizeof(unsigned int)*Np*m_nNeigh);

        m_deficits = 0;
        parallel_for(blocked_range<size_t>(0,Np),
            ComputeNearestNeighbors(m_deficits, (atomic<unsigned int>*)m_neighbor_array.get(), m_box, m_nNeigh, m_rmax, m_lc, r));

        // Increase m_rmax
        if(m_deficits > 0)
            {
            m_rmax *= 1.1;
            m_lc = locality::LinkCell(m_box, m_rmax);
            }
        } while(m_deficits > 0);

    }

void NearestNeighbors::computePy(boost::python::numeric::array r)
    {
    //validate input type and rank
    num_util::check_type(r, PyArray_FLOAT);
    num_util::check_rank(r, 2);

    // validate that the 2nd dimension is only 3 for r and 4 for q
    num_util::check_dim(r, 1, 3);
    unsigned int Np = num_util::shape(r)[0];
    m_Np = Np;

    // get the raw data pointers and compute order parameter
    vec3<float>* r_raw = (vec3<float>*) num_util::data(r);

    // compute the order parameter with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(r_raw, Np);
        }
    }

void export_NearestNeighbors()
    {
    class_<NearestNeighbors>("NearestNeighbors", init<trajectory::Box&, unsigned int, float>())
        .def("getBox", &NearestNeighbors::getBox, return_internal_reference<>())
        .def("getNNeigh", &NearestNeighbors::getNNeigh)
        .def("getRMax", &NearestNeighbors::getRMax)
        .def("getNeighbors", &NearestNeighbors::getNeighborsPy)
        .def("compute", &NearestNeighbors::computePy)
        ;
    }

}; }; // end namespace freud::order
