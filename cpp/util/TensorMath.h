// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include "HOOMDMath.h"

template < class Real >
struct tensor4
    {
    tensor4()
        {
        memset((void*)&data, 0, sizeof(float)*81);
        }
    tensor4(vec3<Real> &_vector)
        {
        unsigned int cnt = 0;
        float v[3];
        v[0] = _vector.x;
        v[1] = _vector.y;
        v[2] = _vector.z;
        for (unsigned int i = 0; i < 3; i++)
            {
            float v_i = v[i];
            for (unsigned int j = 0; j < 3; j++)
                {
                float v_j = v[j];
                for (unsigned int k = 0; k < 3; k++)
                    {
                    float v_k = v[k];
                    for (unsigned int l = 0; l < 3; l++)
                        {
                        float v_l = v[l];
                        data[cnt] = v_i * v_j * v_k * v_l;
                        cnt++;
                        }
                    }
                }
            }
        }
    tensor4(Real (&_data)[81])
        {
        memcpy((void*)data, (void*)_data, sizeof(float)*81);
        }
    tensor4(float* _data)
        {
        memcpy((void*)data, (void*)_data, sizeof(float)*81);
        }
    Real data[81];
    };

template < class Real >
tensor4<Real> operator+(const tensor4<Real>& a, const tensor4<Real>& b)
    {
    tensor4<Real> c;
    for (unsigned int i=0; i<81; i++)
        {
        c.data[i] = a.data[i] + b.data[i];
        }
    return c;
    }

template < class Real >
tensor4<Real> operator+(const tensor4<Real>& a, const Real& b)
    {
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] + b;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator+=(tensor4<Real>& a, const tensor4<Real>& b)
    {
    for (unsigned int i=0; i<81; i++)
        {
        a.data[i] += b.data[i];
        }
    return a;
    }

template < class Real >
tensor4<Real> operator+=(tensor4<Real>& a, const Real& b)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] += b;
        }
    return a;
    }

template < class Real >
tensor4<Real> operator-(const tensor4<Real>& a, const tensor4<Real>& b)
    {
    tensor4<Real> c;
    for (unsigned int i=0; i<81; i++)
        {
        c.data[i] = a.data[i] - b.data[i];
        }
    return c;
    }

template < class Real >
tensor4<Real> operator-(const tensor4<Real>& a, const Real& b)
    {
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] - b;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator-=(tensor4<Real>& a, const tensor4<Real>& b)
    {
    for (unsigned int i=0; i<81; i++)
        {
        a.data[i] -= b.data[i];
        }
    return a;
    }

template < class Real >
tensor4<Real> operator-=(tensor4<Real>& a, const Real& b)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] -= b;
        }
    }

template < class Real >
float dot(const tensor4<Real>& a, const tensor4<Real>& b)
    {
    Real c = 0;
    for (unsigned int i = 0; i < 81; i++)
        {
        c += a.data[i] * b.data[i];
        }
    return c;
    }

template < class Real >
tensor4<Real> operator*(const tensor4<Real>& a, const Real& b)
    {
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] * b;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator/(const tensor4<Real>& a, const Real& b)
    {
    Real b_inv = 1.0/b;
    tensor4<Real> c;
    for (unsigned int i = 0; i < 81; i++)
        {
        c.data[i] = a.data[i] * b_inv;
        }
    return c;
    }

template < class Real >
tensor4<Real> operator*=(tensor4<Real>& a, const Real& b)
    {
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] *= b;
        }
    return a;
    }

template < class Real >
tensor4<Real> operator/=(tensor4<Real>& a, const Real& b)
    {
    Real b_inv = 1.0/b;
    for (unsigned int i = 0; i < 81; i++)
        {
        a.data[i] *= b_inv;
        }
    return a;
    }

#endif // TENSOR_MATH_H
