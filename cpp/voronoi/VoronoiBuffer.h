#include <memory>
#include <vector>

#include "trajectory.h"
#include "Index1D.h"

#ifndef _VoronoiBuffer_H__
#define _VoronoiBuffer_H__

/*! \file VoronoiBuffer.h
    \brief Routines for computing Gaussian smeared densities from points
*/

namespace freud { namespace voronoi {

//! Locates the particles near the border of the box and computes their nearest images to pass to qhull
/*!
*/
class VoronoiBuffer
    {
    public:
        //! Constructor
        VoronoiBuffer(const trajectory::Box& box):m_box(box){}

        //! Get the simulation box
        const trajectory::Box& getBox() const
                {
                return m_box;
                }

        //! Compute the particle images
        void compute(const float3 *points,
                     const unsigned int Np,
                     const float buff);

        std::shared_ptr< std::vector<float3> > getBufferParticles()
            {
            return m_buffer_particles;
            }

        // //!Python wrapper for compute
        // void computePy(boost::python::numeric::array points,
        //                const float buff);

        // //!Python wrapper for getDensity() (returns a copy)
        // boost::python::numeric::array getBufferParticles()
        //         {
        //         std::vector<float3>& buffer_parts = *m_buffer_particles;
        //         std::vector<intp> dims;
        //         float* b = (float*)&buffer_parts[0];
        //         dims.push_back(buffer_parts.size());
        //         if (m_box.is2D())
        //             dims.push_back(2);
        //         else
        //             dims.push_back(3);
        //         return num_util::makeNum(b, dims);
        //         }
    private:
        const trajectory::Box m_box;    //!< Simulation box the particles belong in
        float m_buff;                  //!< Distance from box to duplicate particles
        std::shared_ptr< std::vector<float3> > m_buffer_particles;
    };

}; }; // end namespace freud::density

#endif
