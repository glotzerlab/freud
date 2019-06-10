// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ParticleBuffer.h"

using namespace std;

/*! \file ParticleBuffer.cc
    \brief Replicates particles across periodic boundaries.
*/

namespace freud { namespace box {

void ParticleBuffer::compute(const vec3<float>* points, const unsigned int Np, const vec3<float> buff,
                             const bool use_images)
{
    assert(points);

    if (buff.x < 0)
        throw invalid_argument("Buffer x distance must be non-negative.");
    if (buff.y < 0)
        throw invalid_argument("Buffer y distance must be non-negative.");
    if (buff.z < 0)
        throw invalid_argument("Buffer z distance must be non-negative.");

    m_buffer_particles = std::shared_ptr<std::vector<vec3<float>>>(new std::vector<vec3<float>>());
    m_buffer_ids = std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>());
    std::vector<vec3<float>>& buffer_parts = *m_buffer_particles;
    std::vector<unsigned int>& buffer_ids = *m_buffer_ids;

    // Get the box dimensions
    vec3<float> L(m_box.getL());
    float xy = m_box.getTiltFactorXY();
    float xz = m_box.getTiltFactorXZ();
    float yz = m_box.getTiltFactorYZ();
    bool is2D = m_box.is2D();
    vec3<int> images;

    if (use_images)
    {
        images = vec3<int>(ceil(buff.x), ceil(buff.y), ceil(buff.z));
        m_buffer_box
            = Box((1 + images.x) * L.x, (1 + images.y) * L.y, (1 + images.z) * L.z, xy, xz, yz, is2D);
    }
    else
    {
        images = vec3<int>(ceil(buff.x / L.x), ceil(buff.y / L.y), ceil(buff.z / L.z));
        m_buffer_box = Box(L.x + 2 * buff.x, L.y + 2 * buff.y, L.z + 2 * buff.z, xy, xz, yz, is2D);
    }

    if (is2D)
    {
        L.z = 0;
        xz = 0;
        yz = 0;
        images.z = 0;
    }

    buffer_parts.clear();
    buffer_ids.clear();

    // for each particle
    for (unsigned int particle = 0; particle < Np; particle++)
    {
        for (int i = use_images ? 0 : -images.x; i <= images.x; i++)
        {
            for (int j = use_images ? 0 : -images.y; j <= images.y; j++)
            {
                for (int k = use_images ? 0 : -images.z; k <= images.z; k++)
                {
                    // Skip the origin image
                    if (i == 0 && j == 0 && k == 0)
                    {
                        continue;
                    }

                    // Compute the new position for the buffer particle,
                    // shifted by images.
                    vec3<float> particle_image = points[particle];
                    particle_image += float(i) * m_box.getLatticeVector(0);
                    particle_image += float(j) * m_box.getLatticeVector(1);
                    if (!is2D)
                    {
                        particle_image += float(k) * m_box.getLatticeVector(2);
                    }

                    if (use_images)
                    {
                        // Wrap the positions back into the buffer box and
                        // always append them if a number of images was
                        // specified. Performing the check this way ensures we
                        // have the correct number of particles instead of
                        // relying on the floating point precision of the
                        // fractional check below.
                        buffer_parts.push_back(m_buffer_box.wrap(particle_image));
                        buffer_ids.push_back(particle);
                    }
                    else
                    {
                        // When using a buffer "skin distance," we check the
                        // fractional coordinates to see if the particles are
                        // inside the buffer box. Unexpected results may occur
                        // due to numerical imprecision in this check!
                        vec3<float> buff_frac = m_buffer_box.makeFraction(particle_image);
                        if (0 <= buff_frac.x && buff_frac.x < 1 && 0 <= buff_frac.y && buff_frac.y < 1
                            && (is2D || (0 <= buff_frac.z && buff_frac.z < 1)))
                        {
                            buffer_parts.push_back(particle_image);
                            buffer_ids.push_back(particle);
                        }
                    }
                }
            }
        }
    }
}

}; }; // end namespace freud::box
