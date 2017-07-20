// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "VoronoiBuffer.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <vector>
#include <memory>

using namespace std;

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points
*/

namespace freud { namespace voronoi {

void VoronoiBuffer::compute(const float3 *points,
                            const unsigned int Np,
                            const float buff)
    {
    assert(points);

    m_buffer_particles = std::shared_ptr<std::vector<float3> >(new std::vector<float3>());
    std::vector<float3>& buffer_parts = *m_buffer_particles;
    //get the box dimensions
    float lx = m_box.getLx();
    float ly = m_box.getLy();
    float lz = m_box.getLz();
    float lx_2 = 0.5*lx;
    float ly_2 = 0.5*ly;
    float lz_2 = 0.5*lz;
    float lx_2_buff = 0.5*lx + buff;
    float ly_2_buff = 0.5*ly + buff;
    float lz_2_buff = 0.5*lz + buff;

    float3 img;
    buffer_parts.clear();
    // for each particle
    for (unsigned int particle = 0; particle < Np; particle++)
      {
      // in 2D, only loop over the 0 z plane
      if (m_box.is2D())
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
                  //check to see if this image in within a
                  if(img.x < lx_2_buff && img.x > -lx_2_buff && img.y < ly_2_buff && img.y > -ly_2_buff)
                      {
                      buffer_parts.push_back(img);
                      }
                  }
          }
        }
        }
      else
        {
        //loop over potential images
        for (int i=-1; i<=1; i++)
        {
          for (int j=-1; j<=1; j++)
          {
              for (int k=-1; k<=1; k++)
              {
                if(!(i==0 && j==0 && k==0))
                  {
                  img.x = points[particle].x + i*lx;
                  img.y = points[particle].y + j*ly;
                  img.z = points[particle].z + k*lz;
                  //check to see if this image in within a
                  if(img.x < lx_2_buff && img.x > -lx_2_buff &&
                     img.y < ly_2_buff && img.y > -ly_2_buff &&
                     img.z < lz_2_buff && img.z > -lz_2_buff)
                      {
                      buffer_parts.push_back(img);
                      }
                  }
              }
          }
        }
        }
      }
    }

// void VoronoiBuffer::computePy(boost::python::numeric::array points, const float buff)
//     {
//     // validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers
//     float3* points_raw = (float3*) num_util::data(points);

//       // compute with the GIL released
//       {
//       util::ScopedGILRelease gil;
//       compute(points_raw, Np, buff);
//       }
//     }

// void export_VoronoiBuffer()
//     {
//     class_<VoronoiBuffer>("VoronoiBuffer", init<trajectory::Box&>())
//             .def("getBox", &VoronoiBuffer::getBox, return_internal_reference<>())
//             .def("compute", &VoronoiBuffer::computePy)
//             .def("getBufferParticles", &VoronoiBuffer::getBufferParticles)
//             ;
//     }

}; };
