#include "shapesplit.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;

using namespace tbb;

/*! \file ShapeSplit.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace shapesplit {

ShapeSplit::ShapeSplit()
    : m_box(box::Box()), m_Np(0), m_Nsplit(0)
    {
    }

void ShapeSplit::updateBox(box::Box& box)
    {
    // see if it is different than the current box
    if (m_box != box)
        {
        m_box = box;
        }
    }

void ShapeSplit::compute(const vec3<float> *points,
                    unsigned int Np,
                    const quat<float> *orientations,
                    const vec3<float> *split_points,
                    unsigned int Nsplit)
    {
    // reallocate the output array if it is not the right size
    if (Np != m_Np || Nsplit != m_Nsplit)
        {
        m_split_array = boost::shared_array<float>(new float[Np*Nsplit*3]);
        m_orientation_array = boost::shared_array<float>(new float[Np*Nsplit*4]);
        }
    parallel_for(blocked_range<size_t>(0,Np),
      [=] (const blocked_range<size_t>& r)
      {
      // create Index
      Index3D b_i = Index3D(3, Nsplit, Np);
      Index3D q_i = Index3D(4, Nsplit, Np);
      // for each point
      for (size_t i = r.begin(); i != r.end(); i++)
          {
          vec3<float> point = points[i];
          for (unsigned int j = 0; j < m_Nsplit; j++)
              {
              vec3<float> split_point = point + rotate(orientations[i], split_points[j]);

              split_point = m_box.wrap(split_point);

              m_split_array[b_i(0, j, i)] = split_point.x;
              m_split_array[b_i(1, j, i)] = split_point.y;
              m_split_array[b_i(2, j, i)] = split_point.z;

              m_orientation_array[q_i(0, j, i)] = orientations[i].s;
              m_orientation_array[q_i(1, j, i)] = orientations[i].v.x;
              m_orientation_array[q_i(2, j, i)] = orientations[i].v.z;
              m_orientation_array[q_i(3, j, i)] = orientations[i].v.z;

              }
          } // done looping over reference points
      });

    m_Np = Np;
    m_Nsplit = Nsplit;
    }

}; }; // end namespace freud::shapesplit
