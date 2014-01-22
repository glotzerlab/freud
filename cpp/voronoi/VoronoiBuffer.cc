#include "VoronoiBuffer.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <vector>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace boost::python;

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points
*/

namespace freud { namespace voronoi {

void VoronoiBuffer::compute(const float3 *points,
                            const unsigned int Np,
                            const float buff)
    {
    assert(points);
    
    m_buffer_particles = boost::shared_ptr<std::vector<float3> >(new std::vector<float3>());
    std::vector<float3>& buffer_parts = *m_buffer_particles;
    //get the box dimensions
    float lx = m_box.getLx();
    float ly = m_box.getLy();
    float lz = m_box.getLz();
    
    std::cout<<"box"<<lx<<","<<ly<<","<<lz<<std::endl;
    float3 imgx;
    buffer_parts.clear();
    // for each particle
    for (unsigned int particle = 0; particle < Np; particle++)
      {
      // in 2D, only loop over the 0 z plane
      if (m_box.is2D())
        {
        for (int i=-1; i<=1; i++)
          for (int j=-1; j<=1; j++)
                if(i!=j)
                  {
                  imgx.x = points[particle].x + i*lx;
                  imgx.y = points[particle].y + j*ly;
                  imgx.z = 0.0;
                  //check to see if this image in within a 
                  if( ((imgx.x<0 && imgx.x>-buff) || (imgx.x-lx<buff && imgx.x>lx)) &&
                      ((imgx.y<0 && imgx.y>-buff) || (imgx.y-ly<buff && imgx.y>ly)))
                      {
                      buffer_parts.push_back(imgx);
                      }
                  }
        }
      else
        {
        //loop over potential images
        for (int i=-1; i<=1; i++)
          for (int j=-1; j<=1; j++)
              for (int k=-1; k<=1; k++)
                if(!(i==j && j==k))
                  {
                  std::cout<<"part"<<buff<<" "<<particle<<std::endl;
                  imgx.x = points[particle].x + i*lx;
                  imgx.y = points[particle].y + j*ly;
                  imgx.z = points[particle].z + k*lz;
                  std::cout<<"part"<<particle<<"--->"<<imgx.x<<","<<imgx.y<<","<<imgx.z<<std::endl;

                  //check to see if this image in within a 
                  if( ((imgx.x<0 && imgx.x>-buff) || (imgx.x-lx<buff && imgx.x>lx)) &&
                      ((imgx.y<0 && imgx.y>-buff) || (imgx.y-ly<buff && imgx.y>ly)) &&
                      ((imgx.z<0 && imgx.z>-buff) || (imgx.z-lz<buff && imgx.z>lz)) )
                      {
                      buffer_parts.push_back(imgx);
                      std::cout<<"ADD"<<std::endl;
                      }
                  }
        }
      }
    }

void VoronoiBuffer::computePy(boost::python::numeric::array points, const float buff)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers
    float3* points_raw = (float3*) num_util::data(points);

      // compute with the GIL released
      {
      util::ScopedGILRelease gil;
      compute(points_raw, Np, buff);
      }
    }

void export_VoronoiBuffer()
    {
    class_<VoronoiBuffer>("VoronoiBuffer", init<trajectory::Box&>())
            .def("getBox", &VoronoiBuffer::getBox, return_internal_reference<>())
            .def("compute", &VoronoiBuffer::computePy)
            .def("getBufferParticles", &VoronoiBuffer::getBufferParticles)
            ;
    }

}; };

