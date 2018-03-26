# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from libcpp.string cimport string
from freud.util._Boost cimport shared_array

cdef extern from "box.h" namespace "freud::box":
    cdef cppclass Box:
        Box()
        Box(float, bool)
        Box(float, float, float, bool)
        Box(float, float, float, float, float, float, bool)

        void setL(vec3[float])
        void setL(float, float, float)

        void set2D(bool)
        bool is2D() const

        float getLx() const
        float getLy() const
        float getLz() const

        vec3[float] getL() const
        vec3[float] getLinv() const

        float getTiltFactorXY() const
        float getTiltFactorXZ() const
        float getTiltFactorYZ() const

        float getVolume() const
        vec3[float] makeCoordinates(const vec3[float] &) const
        vec3[float] makeFraction(const vec3[float] &) const
        vec3[int] getImage(const vec3[float] &) const
        vec3[float] getLatticeVector(unsigned int i) const
        vec3[float] wrap(vec3[float] & v) const
        vec3[float] unwrap(vec3[float] &, vec3[int]&)

        vec3[bool] getPeriodic() const
        bool getPeriodicX() const
        bool getPeriodicY() const
        bool getPeriodicZ() const
        void setPeriodic(bool, bool, bool)
        void setPeriodicX(bool)
        void setPeriodicY(bool)
        void setPeriodicZ(bool)
