// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <stdexcept>
#include <vector>

#include "ParticleBuffer.h"

using namespace std;

/*! \file ParticleBuffer.cc
    \brief Replicates particles across periodic boundaries.
*/

namespace freud { namespace box {

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
    int ix, iy, iz;

    if (images)
        {
        ix = ceil(buff);
        iy = ceil(buff);
        iz = ceil(buff);
        int n_images = 1 + ceil(buff);
        m_buffer_box = Box(n_images*lx, n_images*ly, n_images*lz, xy, xz, yz, is2D);
        }
    else
        {
        ix = ceil(buff / lx);
        iy = ceil(buff / ly);
        iz = ceil(buff / lz);
        m_buffer_box = Box(lx+2*buff, ly+2*buff, lz+2*buff, xy, xz, yz, is2D);
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
                        vec3<float> frac = m_box.makeFraction(points[particle]);
                        frac.x += i;
                        frac.y += j;
                        frac.z += k;
                        vec3<float> img = m_box.makeCoordinates(frac);
                        vec3<float> buff_frac = m_buffer_box.makeFraction(img);
                        if (0 <= buff_frac.x && buff_frac.x < 1 &&
                            0 <= buff_frac.y && buff_frac.y < 1 &&
                            (is2D || (0 <= buff_frac.z && buff_frac.z < 1)))
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

}; }; // end namespace freud::box
