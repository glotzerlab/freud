#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _RDF_H__
#define _RDF_H__


//! Computes the RDF (g(r)) for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.
    
    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.
    
    In its current
*/
class RDF
    {
    public:
        //! Constructor
        RDF(const Box& box, float rmax, float dr);
        
        //! Get the simulation box
        const Box& getBox() const
            {
            return m_box;
            }
        
        //! Compute the RDF
        void compute(float *x_ref_data,
                     float *y_ref_data,
                     float *z_ref_data,
                     unsigned int Nref,
                     float *x_data,
                     float *y_data,
                     float *z_data,
                     unsigned int Np);
        
        //! Python wrapper for compute
        void computePy(boost::python::numeric::array x_ref,
                       boost::python::numeric::array y_ref,
                       boost::python::numeric::array z_ref,
                       boost::python::numeric::array x,
                       boost::python::numeric::array y,
                       boost::python::numeric::array z);
                       
        //! Get a reference to the last computed rdf
        boost::shared_array<float> getRDF()
            {
            return m_rdf_array;
            }
        
        //! Get a reference to the r array
        boost::shared_array<float> getR()
            {
            return m_r_array;
            }
        
        //! Python wrapper for getRDF() (returns a copy)
        boost::python::numeric::array getRDFPy()
            {
            float *arr = m_rdf_array.get();
            return num_util::makeNum(arr, m_nbins);
            }

        //! Python wrapper for getR() (returns a copy)
        boost::python::numeric::array getRPy()
            {
            float *arr = m_r_array.get();
            return num_util::makeNum(arr, m_nbins);
            }
    private:
        Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;         //!< Maximum r at which to compute g(r)
        float m_dr;           //!< Step size for r in the computation
        LinkCell m_lc;        //!< LinkCell to bin particles for the computation
        unsigned int m_nbins; //!< Number of r bins to compute g(r) over
        
        boost::shared_array<float> m_rdf_array;         //!< rdf array computed
        boost::shared_array<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        boost::shared_array<float> m_r_array;           //!< array of r values that the rdf is computed at
        boost::shared_array<float> m_vol_array;         //!< array of volumes for each slice of r
    };

//! Exports all classes in this file to python
void export_RDF();

#endif // _TRAJECTORY_H__
