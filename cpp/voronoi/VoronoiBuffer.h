// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>
#include <vector>

#include "box.h"
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
        VoronoiBuffer(const box::Box& box):m_box(box){}

        //! Get the simulation box
        const box::Box& getBox() const
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
        const box::Box m_box;    //!< Simulation box the particles belong in
        std::shared_ptr< std::vector<float3> > m_buffer_particles;
    };

}; }; // end namespace freud::density

#endif
