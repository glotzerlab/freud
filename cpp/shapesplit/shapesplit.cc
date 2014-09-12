#include "shapesplit.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;

using namespace tbb;

/*! \file ShapeSplit.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace shapesplit {

ShapeSplit::ShapeSplit(const trajectory::Box& box)
    : m_box(box), m_Np(0), m_Nsplit(0)
    {
    }

class SplitPoints
    {
    private:
        float *m_split_array;
        const trajectory::Box m_box;
        const vec3<float> *m_points;
        const unsigned int m_Np;
        const vec3<float> *m_split_points;
        const unsigned int m_Nsplit;
    public:
        SplitPoints(float *split_array,
                    const trajectory::Box &box,
                    const vec3<float> *points,
                    unsigned int Np,
                    const vec3<float> *split_points,
                    unsigned int Nsplit)
            : m_split_array(split_array), m_box(box), m_points(points), m_Np(Np), m_split_points(split_points),
              m_Nsplit(Nsplit)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            // create Index
            Index3D b_i = Index3D(3, m_Nsplit, m_Np);
            // for each point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                vec3<float> point = m_points[i];
                for (unsigned int j = 0; j < m_Nsplit; j++)
                    {
                    vec3<float> split_point = point + vec3<float>(m_split_points[j]);

                    float3 wrapped(m_box.wrap(make_float3(split_point.x, split_point.y, split_point.z)));
                    split_point = vec3<float>(wrapped.x, wrapped.y, wrapped.z);

                    m_split_array[b_i(0, j, i)] = split_point.x;
                    m_split_array[b_i(1, j, i)] = split_point.y;
                    m_split_array[b_i(2, j, i)] = split_point.z;

                    }
                } // done looping over reference points
            }
    };

void ShapeSplit::compute(const vec3<float> *points,
                    unsigned int Np,
                    const vec3<float> *split_points,
                    unsigned int Nsplit)
    {
    // reallocate the output array if it is not the right size
    if (Np != m_Np || Nsplit != m_Nsplit)
        {
        m_split_array = boost::shared_array<float>(new float[Np*Nsplit*3]);
        }
    parallel_for(blocked_range<size_t>(0,Np), SplitPoints(m_split_array.get(),
                                                          m_box,
                                                          points,
                                                          Np,
                                                          split_points,
                                                          Nsplit));
    m_Np = Np;
    m_Nsplit = Nsplit;
    }

void ShapeSplit::computePy(boost::python::numeric::array points,
                    boost::python::numeric::array split_points)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(split_points, PyArray_FLOAT);
    num_util::check_rank(split_points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    num_util::check_dim(split_points, 1, 3);
    unsigned int Nsplit = num_util::shape(split_points)[0];

    // get the raw data pointers and compute the cell list
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    vec3<float>* split_points_raw = (vec3<float>*) num_util::data(split_points);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np, split_points_raw, Nsplit);
        }
    }

void export_ShapeSplit()
    {
    class_<ShapeSplit>("ShapeSplit", init<trajectory::Box&>())
        .def("getBox", &ShapeSplit::getBox, return_internal_reference<>())
        .def("compute", &ShapeSplit::computePy)
        .def("getShapeSplit", &ShapeSplit::getShapeSplitPy)
        ;
    }

}; }; // end namespace freud::shapesplit
