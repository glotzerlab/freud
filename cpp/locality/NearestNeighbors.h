#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
// hack to keep VectorMath's swap from polluting the global namespace
// if this is a problem, we need to solve it
#include "VectorMath.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#include "tbb/atomic.h"

#ifndef _NEAREST_NEIGHBORS_H__
#define _NEAREST_NEIGHBORS_H__

/*! \file NearestNeighbors.h
  \brief Find the requested number of nearest neighbors
*/

namespace freud { namespace locality {

/*! Find the requested number of nearest neighbors
*/
class NearestNeighbors
    {
public:
    // Null constructor for use in triclinic; will be removed when cell list is fixed
    NearestNeighbors();
    //! Constructor
    //!
    //! \param box This frame's box
    //! \param rmax Initial guess of the maximum radius to look for n_neigh neighbors
    //! \param nNeigh Number of neighbors to find
    NearestNeighbors(float rmax,
                     unsigned int nNeigh);

    ~NearestNeighbors();

    void setRMax(float rmax)
        {
        m_rmax = rmax;
        m_lc->setCellWidth(m_rmax);
        }

    void setRMaxPy(float rmax)
        {
        m_rmax = rmax;
        m_lc->setCellWidth(m_rmax);
        }

    //! Get the simulation box
    const trajectory::Box& getBox() const
        {
        return m_box;
        }

    //! Get the number of neighbors
    unsigned int getNNeigh() const
        {
        return m_nNeigh;
        }

    //! Get the current cutoff radius used
    float getRMax() const
        {
        return m_rmax;
        }

    //! Get the current cutoff radius used
    float getRMaxPy() const
        {
        return m_rmax;
        }

    //! Get a reference to the neighbors array
    boost::shared_array<unsigned int> getNeighbors(unsigned int i) const
        {
        // create the array
        boost::shared_array<unsigned int> requested_neighbors = boost::shared_array<unsigned int>(new unsigned int[m_nNeigh]);
        // find the position from which to read neighbors
        unsigned int start_idx = i*m_nNeigh;
        for (unsigned int j=0; j<m_nNeigh; j++)
            {
            requested_neighbors[j] = m_neighbor_array[start_idx + j];
            }
        return requested_neighbors;
        }

    //! Python wrapper for getNeighbors() (returns a copy)
    boost::python::numeric::array getNeighborsPy(unsigned int i)
        {
        // create the array
        boost::shared_array<unsigned int> requested_neighbors = boost::shared_array<unsigned int>(new unsigned int[m_nNeigh]);
        // find the position from which to read neighbors
        unsigned int start_idx = i*m_nNeigh;
        for (unsigned int j=0; j<m_nNeigh; j++)
            {
            requested_neighbors[j] = m_neighbor_array[start_idx + j];
            }
        unsigned int *arr = requested_neighbors.get();
        return num_util::makeNum(arr, m_nNeigh);
        }

    //! Get a reference to the neighborlist array
    boost::shared_array<unsigned int> getNeighborList() const
        {
        return m_neighbor_array;
        }

    //! Python wrapper for getNeighbors() (returns a copy)
    boost::python::numeric::array getNeighborListPy()
        {
        unsigned int *arr = m_neighbor_array.get();
        return num_util::makeNum(arr, m_nNeigh*m_Nref);
        }

    //! Get a reference to the distance array
    boost::shared_array<float> getRsq(float i) const
        {
        // create the array
        boost::shared_array<float> requested_rsq = boost::shared_array<float>(new float[m_nNeigh]);
        // find the position from which to read neighbors
        unsigned int start_idx = i*m_nNeigh;
        for (unsigned int j=0; j<m_nNeigh; j++)
            {
            requested_rsq[j] = m_rsq_array[start_idx + j];
            }
        return requested_rsq;
        }

    //! Python wrapper for getR() (returns a copy)
    boost::python::numeric::array getRsqPy(float i)
        {
        // create the array
        boost::shared_array<float> requested_rsq = boost::shared_array<float>(new float[m_nNeigh]);
        // find the position from which to read neighbors
        unsigned int start_idx = i*m_nNeigh;
        for (unsigned int j=0; j<m_nNeigh; j++)
            {
            requested_rsq[j] = m_rsq_array[start_idx + j];
            }
        float *arr = requested_rsq.get();
        return num_util::makeNum(arr, m_nNeigh);
        }

    //! Get a reference to the distanceList array
    boost::shared_array<float> getRsqList() const
        {
        return m_rsq_array;
        }

    //! Python wrapper for getRList() (returns a copy)
    boost::python::numeric::array getRsqListPy()
        {
        float *arr = m_rsq_array.get();
        return num_util::makeNum(arr, m_nNeigh*m_Np);
        }

    //! find the requested nearest neighbors
    void compute(trajectory::Box& box, const vec3<float> *ref_pos, unsigned int Nref, const vec3<float> *pos, unsigned int Np);

    //! Python wrapper for compute
    void computePy(trajectory::Box& box, boost::python::numeric::array ref_pos, boost::python::numeric::array pos);

private:
    trajectory::Box m_box;            //!< Simulation box the particles belong in
    unsigned int m_nNeigh;            //!< Number of neighbors to calculate
    float m_rmax;                     //!< Maximum r at which to determine neighbors
    unsigned int m_Np;                //!< Number of particles for which nearest neighbors checks
    unsigned int m_Nref;                //!< Number of particles for which nearest neighbors calcs
    locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
    tbb::atomic<unsigned int> m_deficits; //!< Neighbor deficit count from the last compute step
    boost::shared_array<unsigned int> m_neighbor_array;         //!< array of nearest neighbors computed
    boost::shared_array<float> m_rsq_array;         //!< array of distances to neighbors
    };

//! Exports all classes in this file to python
void export_NearestNeighbors();

}; }; // end namespace freud::locality

#endif // _NEAREST_NEIGHBORS_H__
