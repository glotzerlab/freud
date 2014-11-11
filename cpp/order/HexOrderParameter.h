#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _HEX_ORDER_PARAMTER_H__
#define _HEX_ORDER_PARAMTER_H__

/*! \file HexOrderParameter.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class HexOrderParameter
    {
    public:
        //! Constructor
        HexOrderParameter(float rmax, float k);

        //! Destructor
        ~HexOrderParameter();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the hex order parameter
        void compute(const vec3<float> *points,
                     unsigned int Np);

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array points);

        //! Get a reference to the last computed psi
        boost::shared_array< std::complex<float> > getPsi()
            {
            return m_psi_array;
            }

        //! Python wrapper for getPsi() (returns a copy)
        boost::python::numeric::array getPsiPy()
            {
            std::complex<float> *arr = m_psi_array.get();
            return num_util::makeNum(arr, m_Np);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;                        //!< Multiplier in the exponent
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_Np;                //!< Last number of points computed

        boost::shared_array< std::complex<float> > m_psi_array;         //!< psi array computed
    };

//! Exports all classes in this file to python
void export_HexOrderParameter();

}; }; // end namespace freud::order

#endif // _HEX_ORDER_PARAMTER_H__
