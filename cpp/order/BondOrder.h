#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _BOND_ORDER_H__
#define _BOND_ORDER_H__

/*! \file BondOrder.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class BondOrder
    {
    public:
        //! Constructor
        BondOrder(float rmax, float k=6, unsigned int n=0);

        //! Destructor
        ~BondOrder();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the bond order array to all zeros
        void resetBondOrder();

        //! Python wrapper for reset method
        void resetBondOrderPy()
            {
            resetBondOrder();
            }

        //! accumulate the bond order
        void accumulate(const vec3<float> *points,
                        unsigned int Np);

        //! Python wrapper for accumulate
        void accumulatePy(trajectory::Box& box,
                          boost::python::numeric::array points);

        //! Compute the bond order
        void compute(const vec3<float> *points,
                     unsigned int Np);

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array points);

        //! Get a reference to the last computed rdf
        boost::shared_array<float> getBondOrder();

        //! Get a reference to the r array
        boost::shared_array<float> getTheta()
            {
            return m_t_array;
            }

        //! Get a reference to the N_r array
        boost::shared_array<float> getPhi()
            {
            return m_p_array;
            }

        //! Python wrapper for getRDF() (returns a copy)
        boost::python::numeric::array getBondOrderPy();

        //! Python wrapper for getR() (returns a copy)
        boost::python::numeric::array getThetaPy()
            {
            float *arr = m_t_array.get();
            return num_util::makeNum(arr, m_nbins_t);
            }

        //! Python wrapper for getNr() (returns a copy)
        boost::python::numeric::array getPhiPy()
            {
            float *arr = m_p_array.get();
            return num_util::makeNum(arr, m_nbins_p);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;                        //!< Multiplier in the exponent
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_Np;                //!< Last number of points computed
        unsigned int m_nbins_t;           //!< number of bins for theta
        unsigned int m_nbins_p;           //!< number of bins for phi

        boost::shared_array<unsigned int> m_bin_counts;         //!< bin counts computed
        boost::shared_array<float> m_bo_array;         //!< bond order array computed
        boost::shared_array<float> m_sa_array;         //!< bond order array computed
        boost::shared_array<float> m_theta_array;         //!< theta array computed
        boost::shared_array<float> m_phi_array;         //!< phi order array computed
    };

//! Exports all classes in this file to python
void export_BondOrder();

}; }; // end namespace freud::order

#endif // _BOND_ORDER_H__
