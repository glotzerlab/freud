// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <stdexcept>
#include <vector>

#include "ParticleBuffer.h"

using namespace std;

/*! \file ParticleBuffer.cc
    \brief Computes a buffer of particles to support wrapped positions in qhull
*/

namespace freud { namespace util {

void ParticleBuffer::compute(const vec3<float> *points,
                            const unsigned int Np,
                            const float buff,
                            const bool images)
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
    float xy = m_box.getTiltFactorXY();
    float xz = m_box.getTiltFactorXZ();
    float yz = m_box.getTiltFactorYZ();
    bool is2D = m_box.is2D();

    float lx_2_buff, ly_2_buff, lz_2_buff;
    int ix, iy, iz;

    if (images)
        {
        ix = ceil(buff);
        iy = ceil(buff);
        iz = ceil(buff);
        lx_2_buff = (ix + 0.5) * lx;
        ly_2_buff = (iy + 0.5) * ly;
        lz_2_buff = (iz + 0.5) * lz;
        }
    else
        {
        ix = ceil(buff / lx);
        iy = ceil(buff / ly);
        iz = ceil(buff / lz);
        lx_2_buff = 0.5*lx + buff;
        ly_2_buff = 0.5*ly + buff;
        lz_2_buff = 0.5*lz + buff;
        }

    if (is2D)
        {
        iz = 0;
        xz = 0;
        yz = 0;
        lz = 0;
        }

    vec3<float> img;
    buffer_parts.clear();
    buffer_ids.clear();

    // for each particle
    for (unsigned int particle = 0; particle < Np; particle++)
        {
        for (int i=-ix; i<=ix; i++)
            {
            for (int j=-iy; j<=iy; j++)
                {
                for (int k=-iz; k<=iz; k++)
                    {
                    if (i != 0 || j != 0 || k != 0)
                        {
                        img.x = points[particle].x + i*lx + j*ly*xy + k*lz*xz;
                        img.y = points[particle].y + j*ly + k*lz*yz;
                        img.z = points[particle].z + k*lz;
                        // Check to see if this image is within the buffer
                        float xadj = img.y*xy + img.z*xz;
                        float yadj = img.z*yz;
                        if (img.x < (lx_2_buff + xadj) && img.x > (-lx_2_buff + xadj) &&
                            img.y < (ly_2_buff + yadj) && img.y > (-ly_2_buff + yadj) &&
                            (is2D || img.z < lz_2_buff && img.z > -lz_2_buff))
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

}; }; // end namespace freud::util
