#include "HOOMDMath.h"
#include "VectorMath.h"
#include "Index1D.h"

#ifndef _TRIANGLES_H__
#define _TRIANGLES_H__

/*! \file triangles.h
    \brief Helper routines for generating triangle geometry
*/

namespace freud { namespace viz {

/*! \internal
    \brief Helper function for triangles conversion
*/
/*
void triangle_rotate(float2 *vert_array,
              const float2 *position_array,
              const float *angle_array,
              const float2 *poly_array,
              unsigned int N);
*/
float4 quat_mult(float4 a, float4 b);
float dot_prod(float4 a, float4 b);
float4 gen_q(float angle);
float4 gen_qs(float angle);
float2 q_rotate(float2 point, float angle);
float2 mat_rotate(float2 point, float angle);
// void triangle_rotatePy(boost::python::numeric::array vert_array,
//                 boost::python::numeric::array color_array,
//                 boost::python::numeric::array position_array,
//                 boost::python::numeric::array angle_array,
//                 boost::python::numeric::array triangle_array,
//                 boost::python::numeric::array poly_colors
//                 );
void triangle_rotate(float2 *vert_array,
              float4 *color_array,
              const float2 *position_array,
              const float *angle_array,
              const float2 *tri_array,
              const float4 *poly_colors,
              unsigned int N,
              unsigned int NT);

// void triangle_rotatePy_mat(boost::python::numeric::array vert_array,
//                 boost::python::numeric::array color_array,
//                 boost::python::numeric::array position_array,
//                 boost::python::numeric::array angle_array,
//                 boost::python::numeric::array triangle_array,
//                 boost::python::numeric::array poly_colors
//                 );
void triangle_rotate_mat(float2 *vert_array,
              float4 *color_array,
              const float2 *position_array,
              const float *angle_array,
              const float2 *tri_array,
              const float4 *poly_colors,
              unsigned int N,
              unsigned int NT);

} } // end namespace freud::viz

#endif
