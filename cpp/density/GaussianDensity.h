#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _GaussianDensity_H__
#define _GaussianDensity_H__

namespace freud { namespace density {

//! Computes the the density of a system on a grid.
/*! Replaces particle positions with a gaussian and calculates the 
        contribution from the grid based upon the the distance of the grid cell
        from the center of the Gaussian. 
*/
class GaussianDensity
    {
    public:
        //! Constructor
        GaussianDensity(const trajectory::Box& box, unsigned int width, float r_cut, float sigma);

        //! Get the simulation box
        const trajectory::Box& getBox() const
                {
                return m_box;
                }

        //! Compute the Density
        void compute(const float3 *points,
                                 unsigned int Np);

        //!Python wrapper for compute
        void computePy(boost::python::numeric::array points);

        //!Get a reference to the last computed Density
        boost::shared_array<float> getDensity()
                {
                return m_Density_array;
                }

        //!Python wrapper for getDensity() (returns a copy)
        boost::python::numeric::array getDensityPy()
                {
                float *arr = m_Density_array.get();
                std::vector<intp> dims;
                dims.push_back(m_width);
                dims.push_back(m_width);
                if (!m_box.is2D())
                    dims.push_back(m_width);
                
                return num_util::makeNum(arr, dims);
                }
    private:
        const trajectory::Box m_box;    //!< Simulation box the particles belong in
        unsigned int m_width;           //!< Num of bins on one side of the cube
        float m_r_cut;                  //!< Max r at which to compute density
        float m_sigma;                  //!< Variance
        Index3D m_bi;                   //!< Bin indexer

        boost::shared_array<float> m_Density_array;            //! computed density array
    };


/*! \internal
    \brief Exports all classes in this file to python
*/
void export_GaussianDensity();

}; }; // end namespace freud::density

#endif _GaussianDensity_H__
