#include <stdexcept>

#include "triangles.h"
#include "ScopedGILRelease.h"

#include <iostream>
#include <math.h>

using namespace std;

/*! \file triangles.cc
    \brief Helper routines for generating triangle geometry
*/

namespace freud { namespace viz {

/*! \internal
    \quaternion multiplication function in c

    \param a Input value: quaternion a (float4)
    \param b Input value: quaternion b (float4)
    \param c Return value: quaternion c (float4)

*/

float4 quat_mult(float4 a, float4 b)
    {
    float4 c;
    c.x = (a.x * b.x) - (a.y * b.y) - (a.z * b.z) - (a.w * b.w);
    c.y = (a.x * b.y) + (a.y * b.x) + (a.z * b.w) - (a.w * b.z);
    c.z = (a.x * b.z) - (a.y * b.w) + (a.z * b.x) + (a.w * b.y);
    c.w = (a.x * b.w) + (a.y * b.z) - (a.z * b.y) + (a.w * b.x);
    return c;
    }

/*! \internal
    \float4 dot product function in c

    \param a Input value: a (float4)
    \param b Input value: b (float4)
    \param c Return value: c (float)

*/

float dot_prod(float4 a, float4 b)
    {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

/*! \internal
    \float4 generates q for quaternion multiplication. Assumes 2D

    \param angle Input value: angle to rotate (float)
    \param q Return value: q (float4)

*/

float4 gen_q(float angle)
    {
    float3 axis;
    float4 q;
    axis.x = 0;
    axis.y = 0;
    axis.z = 1;
    q.x = cosf(0.5 * angle);
    q.y = axis.x * sinf(0.5 * angle);
    q.z = axis.y * sinf(0.5 * angle);
    q.w = axis.z * sinf(0.5 * angle);
    float den = sqrt(dot_prod(q, q));
    q.x = q.x / den;
    q.y = q.y / den;
    q.z = q.z / den;
    q.w = q.w / den;
    return q;
    }

/*! \internal
    \float4 generates qs for quaternion multiplication. Assumes 2D

    \param angle Input value: angle to rotate (float)
    \param qs Return value: qs (float4)

*/

float4 gen_qs(float angle)
    {
    float3 axis;
    float4 q;
    axis.x = 0;
    axis.y = 0;
    axis.z = 1;
    q.x = cosf(0.5 * angle);
    q.y = -1.0 * axis.x * sinf(0.5 * angle);
    q.z = -1.0 * axis.y * sinf(0.5 * angle);
    q.w = -1.0 * axis.z * sinf(0.5 * angle);
    float den = sqrt(dot_prod(q, q));
    q.x = q.x / den;
    q.y = q.y / den;
    q.z = q.z / den;
    q.w = q.w / den;
    return q;
    }

/*! \internal
    \float2 rotation function in c based on quaternions

    \param point Input value: point to be rotated (float2)
    \param angle Input value: angle to be rotated (float)
    \param rot Return value: rot (float2)

*/

float2 q_rotate(float2 point, float angle)
    {

    float4 q = gen_q(angle);
    float4 qs = gen_qs(angle);
    float4 tp;
    tp.x = 0.0;
    tp.y = point.x;
    tp.z = point.y;
    tp.w = 0.0;
    float4 tmp;
    float4 ps;
    tmp = quat_mult(q, tp);
    ps = quat_mult(tmp, qs);
    float2 rot;
    rot.x = ps.y;
    rot.y = ps.z;
    return rot;
    }

/*! \internal
    \float2 rotation function in c based on quaternions

    \param point Input value: point to be rotated (float2)
    \param angle Input value: angle to be rotated (float)
    \param rot Return value: rot (float2)

*/

float2 mat_rotate(float2 point, float angle)
    {
    float2 rot;
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

// void triangle_rotatePy(boost::python::numeric::array vert_array,
//                 boost::python::numeric::array color_array,
//                 boost::python::numeric::array position_array,
//                 boost::python::numeric::array angle_array,
//                 boost::python::numeric::array triangle_array,
//                 boost::python::numeric::array poly_colors
//                 )
//     {
//     //ugh I can't remember which is input and which is output...
//     //validate input type and rank
//     //

//     num_util::check_type(vert_array, NPY_FLOAT);
//     num_util::check_rank(vert_array, 3);

//     // validate that the 2nd dimension is 4
//     num_util::check_dim(vert_array, 2, 2);
//     //unsigned int N = num_util::shape(vert_array)[0];

//     // check that u is consistent
//     num_util::check_type(position_array, NPY_FLOAT);
//     num_util::check_rank(position_array, 2);
//     unsigned int N = num_util::shape(position_array)[0];
//     //if (num_util::shape(position_array)[0] != N)
//     //    throw std::invalid_argument("Input lengths for vert_array and position_array must match");

//     // check that s is consistent
//     num_util::check_type(angle_array, NPY_FLOAT);
//     num_util::check_rank(angle_array, 1);
//     if (num_util::shape(angle_array)[0] != N)
//         throw std::invalid_argument("Input lengths for vert_array and angle_array must match");

//     // check that v is consistent
//     num_util::check_type(triangle_array, NPY_FLOAT);
//     // I think this should be N_Tx3X2
//     num_util::check_rank(triangle_array, 3);
//     unsigned int NT = num_util::shape(triangle_array)[0];

//     num_util::check_type(poly_colors, NPY_FLOAT);
//     // I think this should be N_Tx3X2
//     num_util::check_rank(poly_colors, 2);
//     if (num_util::shape(poly_colors)[0] != N)
//         throw std::invalid_argument("Input lengths for vert_array and poly_colors must match");
//     if (num_util::shape(poly_colors)[1] != 4)
//         throw std::invalid_argument("Input lengths for vert_array and poly_colors must match");

//     // get the raw data pointers and compute conversion
//     float2* vert_array_raw = (float2*) num_util::data(vert_array);
//     float4* color_array_raw = (float4*) num_util::data(color_array);
//     float2* position_array_raw = (float2*) num_util::data(position_array);
//     float* angle_array_raw = (float*) num_util::data(angle_array);
//     float2* triangle_array_raw = (float2*) num_util::data(triangle_array);
//     float4* poly_colors_raw = (float4*) num_util::data(poly_colors);

//         // compute the colormap with the GIL released

//         {
//         util::ScopedGILRelease gil;
//         triangle_rotate(vert_array_raw, color_array_raw, position_array_raw, angle_array_raw, triangle_array_raw, poly_colors_raw, N, NT);
//         }
//     }

// /*! \internal
//     \brief Python wrapper for triangle_rotate

//     \param vert_array Output triangle vertices (N*NTx3x2 float32 array)
//     \param color_array Output triangle colors (N*NTx4 float32 array)
//     \param position_array Input values: positions (N element float32 array)
//     \param angle_array Input values: angles (N element float32 array)
//     \param triangle_array Input values: array of triangle vertices in local coordinates (NTx3x2 element float32 array)
//     \param poly_colors Input polygon color array (Nx4 float32 array)

// */

// void triangle_rotatePy_mat(boost::python::numeric::array vert_array,
//                 boost::python::numeric::array color_array,
//                 boost::python::numeric::array position_array,
//                 boost::python::numeric::array angle_array,
//                 boost::python::numeric::array triangle_array,
//                 boost::python::numeric::array poly_colors
//                 )
//     {
//     //ugh I can't remember which is input and which is output...
//     //validate input type and rank
//     //

//     num_util::check_type(vert_array, NPY_FLOAT);
//     num_util::check_rank(vert_array, 3);

//     // validate that the 2nd dimension is 4
//     num_util::check_dim(vert_array, 2, 2);
//     //unsigned int N = num_util::shape(vert_array)[0];

//     // check that u is consistent
//     num_util::check_type(position_array, NPY_FLOAT);
//     num_util::check_rank(position_array, 2);
//     unsigned int N = num_util::shape(position_array)[0];
//     //if (num_util::shape(position_array)[0] != N)
//     //    throw std::invalid_argument("Input lengths for vert_array and position_array must match");

//     // check that s is consistent
//     num_util::check_type(angle_array, NPY_FLOAT);
//     num_util::check_rank(angle_array, 1);
//     if (num_util::shape(angle_array)[0] != N)
//         throw std::invalid_argument("Input lengths for vert_array and angle_array must match");

//     // check that v is consistent
//     num_util::check_type(triangle_array, NPY_FLOAT);
//     // I think this should be N_Tx3X2
//     num_util::check_rank(triangle_array, 3);
//     unsigned int NT = num_util::shape(triangle_array)[0];

//     num_util::check_type(poly_colors, NPY_FLOAT);
//     // I think this should be N_Tx3X2
//     num_util::check_rank(poly_colors, 2);
//     if (num_util::shape(poly_colors)[0] != N)
//         throw std::invalid_argument("Input lengths for vert_array and poly_colors must match");
//     if (num_util::shape(poly_colors)[1] != 4)
//         throw std::invalid_argument("Input lengths for vert_array and poly_colors must match");

//     // get the raw data pointers and compute conversion
//     float2* vert_array_raw = (float2*) num_util::data(vert_array);
//     float4* color_array_raw = (float4*) num_util::data(color_array);
//     float2* position_array_raw = (float2*) num_util::data(position_array);
//     float* angle_array_raw = (float*) num_util::data(angle_array);
//     float2* triangle_array_raw = (float2*) num_util::data(triangle_array);
//     float4* poly_colors_raw = (float4*) num_util::data(poly_colors);

//         // compute the colormap with the GIL released

//         {
//         util::ScopedGILRelease gil;
//         triangle_rotate_mat(vert_array_raw, color_array_raw, position_array_raw, angle_array_raw, triangle_array_raw, poly_colors_raw, N, NT);
//         }
//     }

/*! \param vert_array Output colormap (Nx4 float32 array)
    \param theta_array Input values: hue angle (N element float32 array)
    \param s_array Input values: saturation (N element float32 array)
    \param v_array Input values: intensity (N element float32 array)
    \param a Alpha value
*/

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

void triangle_rotate(float2 *vert_array,
              float4 *color_array,
              const float2 *position_array,
              const float *angle_array,
              const float2 *tri_array,
              const float4 *poly_colors,
              unsigned int N,
              unsigned int NT)
    {

    // For every polygon aka position

    for (unsigned int i = 0; i < N; i++)
        {
        // for every triangle in that polygon
        for (unsigned int j = 0; j < NT; j++)
            {
                // for every point in that triangle
                for (unsigned int k = 0; k < 3; k++)
                    {
                    // This is the rotated and translated point
                    float2 new_vert;
                    new_vert = q_rotate(tri_array[j * 3 + k], angle_array[i]);
                    new_vert.x = new_vert.x + position_array[i].x;
                    new_vert.y = new_vert.y + position_array[i].y;
                    vert_array[i * NT * 3 + j * 3 + k] = new_vert;
                    }
                //translate
                //put in array
                color_array[i * NT + j] = poly_colors[i];

            }

        }
    }

void triangle_rotate_mat(float2 *vert_array,
              float4 *color_array,
              const float2 *position_array,
              const float *angle_array,
              const float2 *tri_array,
              const float4 *poly_colors,
              unsigned int N,
              unsigned int NT)
    {

    // For every polygon aka position

    for (unsigned int i = 0; i < N; i++)
        {
        // for every triangle in that polygon
        for (unsigned int j = 0; j < NT; j++)
            {
                // for every point in that triangle
                for (unsigned int k = 0; k < 3; k++)
                    {
                    // This is the rotated and translated point
                    float2 new_vert;
                    // Put in for benchmarking
                    // for the love of god take out later

                    new_vert = mat_rotate(tri_array[j * 3 + k], angle_array[i]);
                    new_vert.x = new_vert.x + position_array[i].x;
                    new_vert.y = new_vert.y + position_array[i].y;
                    vert_array[i * NT * 3 + j * 3 + k] = new_vert;
                    }
                //translate
                //put in array
                color_array[i * NT + j] = poly_colors[i];

            }

        }
    }

}; }; // end namespace freud::viz
