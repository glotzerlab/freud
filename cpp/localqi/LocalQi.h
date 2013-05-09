#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include <boost/math/special_functions.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _LOCAL_QI_H__
#define _LOCAL_QI_H__

namespace freud { namespace localqi {

//! Compute the local Steinhardt rotationally invariant Ql order parameter for a set of points
/*! 
*/
class LocalQi
    {
    public:
        //! Constructor for class
        LocalQi(const trajectory::Box& box, float rmax, unsigned int l);
        
        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }
        
        //! Compute the local rotationally invariant Ql order parameter
        void compute(const float3 *points,
                     unsigned int Np);
        
        //! Python wrapper for compute
        void computePy(boost::python::numeric::array points);
                       
        //! Get a reference to the last computed psi
        boost::shared_array< double > getQli()
            {
            return m_Qli;
            }
        
        //! Python wrapper for getPsi() (returns a copy)
        boost::python::numeric::array getQliPy()
            {
            double *arr = m_Qli.get();
            return num_util::makeNum(arr, m_Np);
            }
        
        void Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y);
        
    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        locality::LinkCell m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_l;                   //!< Spherical harmonic l value.  
        unsigned int m_Np;                //!< Last number of points computed
        boost::shared_array< std::complex<double> > m_Qlmi;        //!  Qlm for each particle i
        boost::shared_array< double > m_Qli;         //!< Ql locally invariant order parameter for each particle i;
    };

//! Exports all classes in this file to python
void export_LocalQi();

}; }; // end namespace freud::localqi

#endif // #define _LOCAL_QI_H__
