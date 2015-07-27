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

#ifndef _TRANS_ORDER_PARAMTER_H__
#define _TRANS_ORDER_PARAMTER_H__

/*! \file TransOrderParameter.h
    \brief Compute the translational order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the translational order parameter for a set of points
/*!
*/
class TransOrderParameter
    {
    public:
        //! Constructor
        TransOrderParameter(float rmax, float k=6, unsigned int n=0);

        //! Destructor
        ~TransOrderParameter();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the translational order parameter
        void compute(const vec3<float> *points,
                     unsigned int Np);

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array points);

        //! Get a reference to the last computed dr
        boost::shared_array< std::complex<float> > getDr()
            {
            return m_dr_array;
            }

        //! Python wrapper for getDr() (returns a copy)
        boost::python::numeric::array getDrPy()
            {
            std::complex<float> *arr = m_dr_array.get();
            return num_util::makeNum(arr, m_Np);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;                        //!< Multiplier in the exponent
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_Np;                //!< Last number of points computed

        boost::shared_array< std::complex<float> > m_dr_array;         //!< dr array computed
    };

//! Exports all classes in this file to python
void export_TransOrderParameter();

}; }; // end namespace freud::order

#endif // _TRANS_ORDER_PARAMTER_H__
