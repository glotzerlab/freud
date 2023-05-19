# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool

from freud.util cimport vec3

ctypedef unsigned int uint

cdef extern from "Box.h" namespace "freud::box":
    cdef cppclass Box:
        Box()
        Box(float, bool)
        Box(float, float, float, bool)
        Box(float, float, float, float, float, float, bool)

        bool operator==(const Box &) const
        bool operator!=(const Box &) const

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

        void setTiltFactorXY(float)
        void setTiltFactorXZ(float)
        void setTiltFactorYZ(float)

        float getVolume() const
        void makeAbsolute(const vec3[float]*, unsigned int, vec3[float]*) const
        void makeFractional(const vec3[float]*, unsigned int, vec3[float]*) const
        void getImages(vec3[float]*, unsigned int, vec3[int]*) const
        void wrap(const vec3[float]*, unsigned int, vec3[float]*) const
        void unwrap(const vec3[float]*, const vec3[int]*,
                    unsigned int, vec3[float]*) const
        vec3[float] centerOfMass(vec3[float]*, size_t, float*) const
        void center(vec3[float]*, size_t, float*) const
        void computeDistances(vec3[float]*, unsigned int,
                              vec3[float]*, unsigned int, float*
                              ) except +
        void computeAllDistances(vec3[float]*, unsigned int,
                                 vec3[float]*, unsigned int, float*)
        void contains(vec3[float]*, unsigned int, bool*) const
        vec3[bool] getPeriodic() const
        bool getPeriodicX() const
        bool getPeriodicY() const
        bool getPeriodicZ() const
        void setPeriodic(bool, bool, bool)
        void setPeriodicX(bool)
        void setPeriodicY(bool)
        void setPeriodicZ(bool)
