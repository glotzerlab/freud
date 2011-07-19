#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _HEX_ORDER_PARAMTER_H__
#define _HEX_ORDER_PARAMTER_H__

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*! 
*/
class HexOrderParameter
    {
    public:
        //! Constructor
        HexOrderParameter(const trajectory::Box& box, float rmax);
        
        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }
        
        //! Compute the hex order parameter
        void compute(const float3 *points,
                     unsigned int Np);
        
        //! Python wrapper for compute
        void computePy(boost::python::numeric::array points);
                       
        //! Get a reference to the last computed psi
        boost::shared_array< std::complex<double> > getPsi()
            {
            return m_psi_array;
            }
        
        //! Python wrapper for getPsi() (returns a copy)
        boost::python::numeric::array getPsiPy()
            {
            std::complex<double> *arr = m_psi_array.get();
            return num_util::makeNum(arr, m_Np);
            }

    private:
        Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;         //!< Maximum r at which to determine neighbors
        LinkCell m_lc;        //!< LinkCell to bin particles for the computation
        unsigned int m_Np;    //!< Last number of points computed
        
        boost::shared_array< std::complex<double> > m_psi_array;         //!< psi array computed
    };

//! Exports all classes in this file to python
void export_HexOrderParameter();

}; }; // end namespace freud::order

#endif // _HEX_ORDER_PARAMTER_H__
