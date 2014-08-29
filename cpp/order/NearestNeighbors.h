#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
// hack to keep VectorMath's swap from polluting the global namespace
// if this is a problem, we need to solve it
#define swap freud_swap
#include "VectorMath.h"
#undef swap
#include "num_util.h"
#include "trajectory.h"

#include "tbb/atomic.h"

#ifndef _NEAREST_NEIGHBORS_H__
#define _NEAREST_NEIGHBORS_H__

/*! \file NearestNeighbors.h
  \brief Find the requested number of nearest neighbors
*/

namespace freud { namespace order {

/*! Find the requested number of nearest neighbors
*/
class NearestNeighbors
    {
public:
    //! Constructor
    //!
    //! \param box This frame's box
    //! \param nNeigh Number of neighbors to find
    //! \param rmax Initial guess of the maximum radius to look for n_neigh neighbors
    NearestNeighbors(const trajectory::Box& box, unsigned int nNeigh, float rmax);

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

    //! Get a reference to the x array
    boost::shared_array<unsigned int> getNeighbors()
        {
        return m_neighbor_array;
        }

    //! Python wrapper for getNeighbors() (returns a copy)
    boost::python::numeric::array getNeighborsPy()
        {
        unsigned int *arr = m_neighbor_array.get();
        return num_util::makeNum(arr, m_nNeigh*m_Np);
        }

    //! find the requested nearest neighbors
    void compute(const vec3<float> *r, unsigned int Np);

    //! Python wrapper for compute
    void computePy(boost::python::numeric::array r);

private:
    trajectory::Box m_box;            //!< Simulation box the particles belong in
    unsigned int m_nNeigh;            //!< Number of neighbors to calculate
    float m_rmax;                     //!< Maximum r at which to determine neighbors
    unsigned int m_Np;                //!< Number of particles for which nearest neighbors calc'd
    locality::LinkCell m_lc;          //!< LinkCell to bin particles for the computation
    tbb::atomic<unsigned int> m_deficits; //!< Neighbor deficit count from the last compute step
    boost::shared_array<unsigned int> m_neighbor_array;         //!< array of nearest neighbors computed
    };

//! Exports all classes in this file to python
void export_NearestNeighbors();

}; }; // end namespace freud::order

#endif // _NEAREST_NEIGHBORS_H__
