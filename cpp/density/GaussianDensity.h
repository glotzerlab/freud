#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _GaussianDensity_H__
#define _GaussianDensity_H__

/*! \file GaussianDensity.h
    \brief Routines for computing Gaussian smeared densities from points
*/

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
        GaussianDensity(const trajectory::Box& box, unsigned int width_x, unsigned int width_y, unsigned int width_z,
                        float r_cut, float sigma);

        //! Get the simulation box
        const trajectory::Box& getBox() const
                {
                return m_box;
                }

        //! Compute the Density
        // void compute(const float3 *points,
        //                          unsigned int Np);
        void compute(const vec3<float> *points,
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
                if (!m_box.is2D())
                    dims.push_back(m_width_z);
                dims.push_back(m_width_y);
                dims.push_back(m_width_x);

                return num_util::makeNum(arr, dims);
                }
    private:
        const trajectory::Box m_box;    //!< Simulation box the particles belong in
        unsigned int m_width_x,m_width_y,m_width_z;           //!< Num of bins on one side of the cube
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

#endif 
