// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <stdexcept>
#include <vector>

#include "VoronoiBuffer.h"

using namespace std;

/*! \file VoronoiBuffer.cc
    \brief Computes a buffer of particles to support wrapped positions in qhull
*/

namespace freud { namespace voronoi {

void VoronoiBuffer::compute(const vec3<float> *points,
                            const unsigned int Np,
                            const float buff)
    {
    assert(points);

    m_buffer_particles = std::shared_ptr<std::vector< vec3<float> > >(
            new std::vector< vec3<float> >());
    m_buffer_ids = std::shared_ptr<std::vector< unsigned int > >(
            new std::vector< unsigned int >());
    std::vector< vec3<float> >& buffer_parts = *m_buffer_particles;
    std::vector< unsigned int >& buffer_ids = *m_buffer_ids;

    // Get the box dimensions
    float lx = m_box.getLx();
    float ly = m_box.getLy();
    float lz = m_box.getLz();
    float lx_2_buff = 0.5*lx + buff;
    float ly_2_buff = 0.5*ly + buff;
    float lz_2_buff = 0.5*lz + buff;
    float ix = ceil(buff / lx);

    vec3<float> img;
    buffer_parts.clear();
    buffer_ids.clear();

    // for each particle
    for (unsigned int particle = 0; particle < Np; particle++)
        {
        // in 2D, only loop over the 0 z plane
      /*  if (m_box.is2D())
            {
            for (int i=-1; i<=1; i++)
                {
                for (int j=-1; j<=1; j++)
                    {
                    if(i != 0 || j != 0)
                        {
                        img.x = points[particle].x + i*lx;
                        img.y = points[particle].y + j*ly;
                        img.z = 0.0;
                        // Check to see if this image is within the buffer
                        if(img.x < lx_2_buff && img.x > -lx_2_buff &&
                           img.y < ly_2_buff && img.y > -ly_2_buff)
                            {
                            buffer_parts.push_back(img);
                            buffer_ids.push_back(particle);
                            }
                        }
                    }
                }
            }
        else
            {
            // Loop over potential images
       */   for (int i=-ix; i<=ix; i++)
                {
                for (int j=-1; j<=1; j++)
                    {
                    if (m_box.is2D())
                        {
                        if(i != 0 || j != 0)
                            {
                            img.x = points[particle].x + i*lx;
                            img.y = points[particle].y + j*ly;
                            img.z = 0.0;
                            // Check to see if this image is within the buffer
                            if(img.x < lx_2_buff && img.x > -lx_2_buff &&
                               img.y < ly_2_buff && img.y > -ly_2_buff)
                                {
                                buffer_parts.push_back(img);
                                buffer_ids.push_back(particle);
                                }
                            }
                        }
                    else
                        {
                        for (int k=-1; k<=1; k++)
                            {
                            if(!(i==0 && j==0 && k==0))
                                {
                                img.x = points[particle].x + i*lx;
                                img.y = points[particle].y + j*ly;
                                img.z = points[particle].z + k*lz;
                                // Check to see if this image is within the buffer
                                if(img.x < lx_2_buff && img.x > -lx_2_buff &&
                                   img.y < ly_2_buff && img.y > -ly_2_buff &&
                                   img.z < lz_2_buff && img.z > -lz_2_buff)
                                    {
                                    buffer_parts.push_back(img);
                                    buffer_ids.push_back(particle);
                                    }
                                }
                            }
                        }
                    }
                }
            }
       // }
    }

}; }; // end namespace freud::voronoi
