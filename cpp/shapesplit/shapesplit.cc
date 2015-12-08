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
    : m_box(trajectory::Box()), m_Np(0), m_Nsplit(0)
    {
    }

void ShapeSplit::updateBox(trajectory::Box& box)
    {
    // see if it is different than the current box
    if (m_box != box)
        {
        m_box = box;
        }
    }

class SplitPoints
    {
    private:
        float *m_split_array;
        float *m_orientation_array;
        const trajectory::Box m_box;
        const vec3<float> *m_points;
        const unsigned int m_Np;
        const quat<float> *m_orientations;
        const vec3<float> *m_split_points;
        const unsigned int m_Nsplit;
    public:
        SplitPoints(float *split_array,
                    float *orientation_array,
                    const trajectory::Box &box,
                    const vec3<float> *points,
                    unsigned int Np,
                    const quat<float> *orientations,
                    const vec3<float> *split_points,
                    unsigned int Nsplit)
            : m_split_array(split_array), m_orientation_array(orientation_array), m_box(box), m_points(points),
              m_Np(Np), m_orientations(orientations), m_split_points(split_points), m_Nsplit(Nsplit)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            // create Index
            Index3D b_i = Index3D(3, m_Nsplit, m_Np);
            Index3D q_i = Index3D(4, m_Nsplit, m_Np);
            // for each point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                vec3<float> point = m_points[i];
                for (unsigned int j = 0; j < m_Nsplit; j++)
                    {
                    vec3<float> split_point = point + rotate(m_orientations[i], m_split_points[j]);

                    split_point = m_box.wrap(split_point);

                    m_split_array[b_i(0, j, i)] = split_point.x;
                    m_split_array[b_i(1, j, i)] = split_point.y;
                    m_split_array[b_i(2, j, i)] = split_point.z;

                    m_orientation_array[q_i(0, j, i)] = m_orientations[i].s;
                    m_orientation_array[q_i(1, j, i)] = m_orientations[i].v.x;
                    m_orientation_array[q_i(2, j, i)] = m_orientations[i].v.z;
                    m_orientation_array[q_i(3, j, i)] = m_orientations[i].v.z;

                    }
                } // done looping over reference points
            }
    };

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
    parallel_for(blocked_range<size_t>(0,Np), SplitPoints(m_split_array.get(),
                                                          m_orientation_array.get(),
                                                          m_box,
                                                          points,
                                                          Np,
                                                          orientations,
                                                          split_points,
                                                          Nsplit));
    m_Np = Np;
    m_Nsplit = Nsplit;
    }

// void ShapeSplit::computePy(trajectory::Box& box,
//                            boost::python::numeric::array points,
//                            boost::python::numeric::array orientations,
//                            boost::python::numeric::array split_points)
//     {
//     // validate input type and rank
//     updateBox(box);
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);
//     num_util::check_type(split_points, NPY_FLOAT);
//     num_util::check_rank(split_points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     num_util::check_dim(split_points, 1, 3);
//     unsigned int Nsplit = num_util::shape(split_points)[0];

//     // get the raw data pointers and compute the cell list
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     vec3<float>* split_points_raw = (vec3<float>*) num_util::data(split_points);

//     // needs to check how many dims there are
//     if (num_util::rank(orientations) == 1)
//         {
//         float *theta_raw = (float*) num_util::data(orientations);
//         quat<float> *orientations_raw = new quat<float>[Np];
//         for (unsigned int i=0; i<Np; i++)
//             {
//             float theta = theta_raw[i];
//             orientations_raw[i] = quat<float>::fromAxisAngle(vec3<float>(0, 0, 1), theta);
//             }
//         // compute with the GIL released
//             {
//             util::ScopedGILRelease gil;
//             compute(points_raw, Np, orientations_raw, split_points_raw, Nsplit);
//             }
//         }
//     else
//         {
//         quat<float>* orientations_raw = (quat<float>*) num_util::data(orientations);
//         // compute with the GIL released
//             {
//             util::ScopedGILRelease gil;
//             compute(points_raw, Np, orientations_raw, split_points_raw, Nsplit);
//             }
//         }

//     }

// void export_ShapeSplit()
//     {
//     class_<ShapeSplit>("ShapeSplit", init<>())
//         .def("getBox", &ShapeSplit::getBox, return_internal_reference<>())
//         .def("compute", &ShapeSplit::computePy)
//         .def("getShapeSplit", &ShapeSplit::getShapeSplitPy)
//         .def("getShapeOrientations", &ShapeSplit::getShapeOrientationsPy)
//         ;
//     }

}; }; // end namespace freud::shapesplit
