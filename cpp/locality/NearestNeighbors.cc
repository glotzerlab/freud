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

/*! \file NearestNeighbors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace locality {

// stop using
NearestNeighbors::NearestNeighbors():
    m_box(trajectory::Box()), m_rmax(0), m_nNeigh(0), m_Np(0), m_deficits()
    {
    m_lc = new locality::LinkCell();
    m_deficits = 0;
    }

NearestNeighbors::NearestNeighbors(float rmax,
                                   unsigned int nNeigh):
    m_box(trajectory::Box()), m_rmax(rmax), m_nNeigh(nNeigh), m_Np(0), m_deficits()
    {
    m_lc = new locality::LinkCell(m_box, m_rmax);
    m_deficits = 0;
    }

NearestNeighbors::~NearestNeighbors()
    {
    delete m_lc;
    }

//! Utility function to sort a pair<float, unsigned int> on the first
//! element of the pair
bool compareRsqVectors(const pair<float, unsigned int> &left,
                       const pair<float, unsigned int> &right)
    {
    return left.first < right.first;
    }

class ComputeNearestNeighbors
    {
private:
    atomic<unsigned int> &m_deficits;
    atomic<float> *m_rsq_array;
    atomic<unsigned int> *m_neighbor_array;
    const trajectory::Box& m_box;
    const unsigned int m_Np;
    const unsigned int m_nNeigh;
    const float m_rmax;
    const locality::LinkCell* m_lc;
    const vec3<float> *m_ref_pos;
    const vec3<float> *m_pos;
public:
    ComputeNearestNeighbors(atomic<unsigned int> &deficits,
                            atomic<float> *r_array,
                            atomic<unsigned int> *neighbor_array,
                            const trajectory::Box& box,
                            const unsigned int Np,
                            const unsigned int nNeigh,
                            const float rmax,
                            const locality::LinkCell* lc,
                            const vec3<float> *ref_pos,
                            const vec3<float> *pos):
        m_deficits(deficits), m_rsq_array(r_array), m_neighbor_array(neighbor_array), m_box(box), m_Np(Np), m_nNeigh(nNeigh), m_rmax(rmax), m_lc(lc),
        m_ref_pos(ref_pos), m_pos(pos)
        {
        }

    void operator()( const blocked_range<size_t>& r ) const
        {
        float rmaxsq = m_rmax * m_rmax;
        // tuple<> is c++11, so for now just make a pair with pairs inside
        // this data structure holds rsq, idx
        vector< pair<float, unsigned int> > neighbors;
        Index2D b_i = Index2D(m_nNeigh, m_Np);
        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            neighbors.clear();

            //get cell point is in
            vec3<float> posi = m_ref_pos[i];
            unsigned int ref_cell = m_lc->getCell(posi);
            unsigned int num_adjacent = 0;

            //loop over neighboring cells
            const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
            for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                {
                unsigned int neigh_cell = neigh_cells[neigh_idx];

                //iterate over particles in cell
                locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                    {

                    //compute r between the two particles
                    vec3<float>rij = m_box.wrap(m_pos[j] - posi);
                    const float rsq(dot(rij, rij));

                    // adds all neighbors within rsq to list of possible neighbors
                    if ((rsq < rmaxsq) && (i != j))
                        {
                        neighbors.push_back(pair<float, unsigned int>(rsq, j));
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
                    m_rsq_array[i*m_nNeigh + k] = neighbors[k].first;
                    m_rsq_array[b_i(k, i)] = neighbors[k].first;
                    m_neighbor_array[b_i(k, i)] = neighbors[k].second;
                    }
                }
            }
        }
    };

void NearestNeighbors::compute(trajectory::Box& box,
                               const vec3<float> *ref_pos,
                               unsigned int Nref,
                               const vec3<float> *pos,
                               unsigned int Np)
    {
    m_box = box;
    // reallocate the output array if it is not the right size
    if (Nref != m_Nref)
        {
        m_rsq_array = boost::shared_array<float>(new float[Nref * m_nNeigh]);
        m_neighbor_array = boost::shared_array<unsigned int>(new unsigned int[Nref * m_nNeigh]);
        }
    // find the nearest neighbors
    do
        {
        // compute the cell list
        m_lc->computeCellList(m_box, pos, Np);

        m_deficits = 0;
        parallel_for(blocked_range<size_t>(0,Np),
            ComputeNearestNeighbors(m_deficits,
                                    (atomic<float>*)m_rsq_array.get(),
                                    (atomic<unsigned int>*)m_neighbor_array.get(),
                                    m_box,
                                    m_Np,
                                    m_nNeigh,
                                    m_rmax,
                                    m_lc,
                                    ref_pos,
                                    pos));

        // Increase m_rmax
        if(m_deficits > 0)
            {
            m_rmax *= 1.1;
            m_lc->setCellWidth(m_rmax);
            }
        } while(m_deficits > 0);
    // save the last computed number of particles
    m_Nref = Nref;
    m_Np = Np;
    }

void NearestNeighbors::computePy(trajectory::Box& box,
                                 boost::python::numeric::array ref_pos,
                                 boost::python::numeric::array pos)
    {
    //validate input type and rank
    num_util::check_type(ref_pos, NPY_FLOAT);
    num_util::check_rank(ref_pos, 2);
    num_util::check_type(pos, NPY_FLOAT);
    num_util::check_rank(pos, 2);

    // validate that the 2nd dimension is only 3 for r and 4 for q
    num_util::check_dim(ref_pos, 1, 3);
    unsigned int Nref = num_util::shape(ref_pos)[0];
    num_util::check_dim(pos, 1, 3);
    unsigned int Np = num_util::shape(pos)[0];

    // get the raw data pointers and compute order parameter
    vec3<float>* ref_pos_raw = (vec3<float>*) num_util::data(ref_pos);
    vec3<float>* pos_raw = (vec3<float>*) num_util::data(pos);

    // compute the order parameter with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(box, ref_pos_raw, Nref, pos_raw, Np);
        }
    }

void export_NearestNeighbors()
    {
    class_<NearestNeighbors>("NearestNeighbors", init<float, unsigned int>())
        .def("getBox", &NearestNeighbors::getBox, return_internal_reference<>())
        .def("getNNeigh", &NearestNeighbors::getNNeigh)
        .def("setRMax", &NearestNeighbors::setRMaxPy)
        .def("getRMax", &NearestNeighbors::getRMaxPy)
        .def("getNeighbors", &NearestNeighbors::getNeighborsPy)
        .def("getNeighborList", &NearestNeighbors::getNeighborListPy)
        .def("getRsq", &NearestNeighbors::getRsqPy)
        .def("getRsqList", &NearestNeighbors::getRsqListPy)
        .def("compute", &NearestNeighbors::computePy)
        ;
    }

}; }; // end namespace freud::locality
