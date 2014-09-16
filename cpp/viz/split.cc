#include <boost/python.hpp>
#include <stdexcept>

#include "num_util.h"
#include "split.h"
#include "ScopedGILRelease.h"

#include <iostream>
#include <math.h>

using namespace std;
using namespace boost::python;

/*! \file split.cc
    \brief Helper routines for splitting particles
*/

namespace freud { namespace viz {

/*! \internal
    \float2 rotation function in c based on rotation matrix

    \param point Input value: point to be rotated (float2)
    \param angle Input value: angle to be rotated (float)
    \param rot Return value: rot (float2)

*/

// float2 rotate(float2 point, float angle)
//     {
//     float2 rot;
//     float mysin = sinf(angle);
//     float mycos = cosf(angle);
//     rot.x = mycos * point.x + -mysin * point.y;
//     rot.y = mysin * point.x + mycos * point.y;
//     return rot;
//     }
vec2<float> rotate(vec2<float> point, float angle)
    {
    vec2<float> rot;
    float mysin = sinf(angle);
    float mycos = cosf(angle);
    rot.x = mycos * point.x + -mysin * point.y;
    rot.y = mysin * point.x + mycos * point.y;
    return rot;
    }

/*! \internal
    \brief Python wrapper for triangle_rotate

    \param vert_array Output triangle vertices (N*NTx3x2 float32 array)
    \param color_array Output triangle colors (N*NTx4 float32 array)
    \param position_array Input values: positions (N element float32 array)
    \param angle_array Input values: angles (N element float32 array)
    \param triangle_array Input values: array of triangle vertices in local coordinates (NTx3x2 element float32 array)
    \param poly_colors Input polygon color array (Nx4 float32 array)

*/

void splitPy(boost::python::numeric::array split_array,
                boost::python::numeric::array sangle_array,
                boost::python::numeric::array position_array,
                boost::python::numeric::array angle_array,
                boost::python::numeric::array centers_array
                )
    {
    //ugh I can't remember which is input and which is output...
    //validate input type and rank
    //

    num_util::check_type(split_array, PyArray_FLOAT);
    num_util::check_rank(split_array, 2);

    num_util::check_type(sangle_array, PyArray_FLOAT);
    num_util::check_rank(sangle_array, 1);

    // validate that the 2nd dimension is 2
    num_util::check_dim(split_array, 1, 3);

    num_util::check_type(position_array, PyArray_FLOAT);
    num_util::check_rank(position_array, 2);
    unsigned int N = num_util::shape(position_array)[0];

    num_util::check_type(angle_array, PyArray_FLOAT);
    num_util::check_rank(angle_array, 1);
    if (num_util::shape(angle_array)[0] != N)
        throw std::invalid_argument("Input lengths for vert_array and angle_array must match");

    num_util::check_type(centers_array, PyArray_FLOAT);
    num_util::check_rank(centers_array, 2);
    unsigned int NS = num_util::shape(centers_array)[0];

    // get the raw data pointers and compute conversion
    // float3* split_array_raw = (float3*) num_util::data(split_array);
    vec3<float>* split_array_raw = (vec3<float>*) num_util::data(split_array);
    float* sangle_array_raw = (float*) num_util::data(sangle_array);
    // float3* position_array_raw = (float3*) num_util::data(position_array);
    vec3<float>* position_array_raw = (vec3<float>*) num_util::data(position_array);
    float* angle_array_raw = (float*) num_util::data(angle_array);
    // float2* centers_array_raw = (float2*) num_util::data(centers_array);
    vec2<float>* centers_array_raw = (vec2<float>*) num_util::data(centers_array);

        {
        util::ScopedGILRelease gil;
        split(split_array_raw, sangle_array_raw, position_array_raw, angle_array_raw, centers_array_raw, N, NS);
        }
    }

/*! \internal
    \brief Python wrapper for triangle_rotate

    \param vert_array Output triangle vertices (N*NTx3x2 float32 array)
    \param color_array Output triangle colors (N*NTx4 float32 array)
    \param position_array Input values: positions (N element float32 array)
    \param angle_array Input values: angles (N element float32 array)
    \param tri_array Input values: array of triangle vertices in local coordinates (NTx3x2 element float32 array)
    \param poly_colors Input polygon color array (Nx4 float32 array)
    \param N Input: number of polygons (unsigned int)
    \param NT Input: number of triangles per polygon (unsigned int)

*/

// void split(float3 *split_array,
//               float *sangle_array,
//               const float3 *position_array,
//               const float *angle_array,
//               const float2 *centers_array,
//               unsigned int N,
//               unsigned int NS)
//     {

//     // For every polygon aka position

//     for (unsigned int i = 0; i < N; i++)
//         {
//         // for every center in that polygon
//         for (unsigned int j = 0; j < NS; j++)
//             {
//                 // This is the rotated and translated center
//                 float2 new_center;
//                 new_center = rotate(centers_array[j], angle_array[i]);
//                 float3 new_pos;
//                 new_pos.x = new_center.x + position_array[i].x;
//                 new_pos.y = new_center.y + position_array[i].y;
//                 new_pos.z = position_array[i].z;
//                 split_array[i * NS + j] = new_pos;
//                 sangle_array[i * NS + j] = angle_array[i];
//             }

//         }
//     }

void split(vec3<float> *split_array,
              float *sangle_array,
              const vec3<float> *position_array,
              const float *angle_array,
              const vec2<float> *centers_array,
              unsigned int N,
              unsigned int NS)
    {

    // For every polygon aka position

    for (unsigned int i = 0; i < N; i++)
        {
        // for every center in that polygon
        for (unsigned int j = 0; j < NS; j++)
            {
                // This is the rotated and translated center
                vec2<float> new_center;
                new_center = rotate(centers_array[j], angle_array[i]);
                vec3<float> new_pos;
                new_pos.x = new_center.x + position_array[i].x;
                new_pos.y = new_center.y + position_array[i].y;
                new_pos.z = position_array[i].z;
                split_array[i * NS + j] = new_pos;
                sangle_array[i * NS + j] = angle_array[i];
            }

        }
    }

void export_split()
    {
    def("split", &splitPy);
    }

}; }; // end namespace freud::viz
