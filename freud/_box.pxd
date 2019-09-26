# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from freud.util cimport vec3
from libcpp.vector cimport vector
from libcpp.string cimport string

ctypedef unsigned int uint

cdef extern from "Box.h" namespace "freud::box":
    cdef cppclass Box:
        Box()
        Box(float, bool_t)
        Box(float, float, float, bool_t)
        Box(float, float, float, float, float, float, bool_t)

        void setL(vec3[float])
        void setL(float, float, float)

        void set2D(bool_t)
        bool_t is2D() const

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
        void makeAbsolute(vec3[float]*, unsigned int) except +
        void makeFractional(vec3[float]*, unsigned int) except +
        void getImage(vec3[float]*, unsigned int, vec3[int]*) except +
        vec3[float] getLatticeVector(unsigned int i) except +
        void wrap(vec3[float]* vs, unsigned int Nv) except +
        void unwrap(vec3[float]*, const vec3[int]*,
                    unsigned int) except +

        vec3[bool_t] getPeriodic() const
        bool_t getPeriodicX() const
        bool_t getPeriodicY() const
        bool_t getPeriodicZ() const
        void setPeriodic(bool_t, bool_t, bool_t)
        void setPeriodicX(bool_t)
        void setPeriodicY(bool_t)
        void setPeriodicZ(bool_t)


cdef extern from "PeriodicBuffer.h" namespace "freud::box":
    cdef cppclass PeriodicBuffer:
        PeriodicBuffer(const Box &)
        const Box & getBox() const
        const Box & getBufferBox() const
        void compute(
            const vec3[float]*,
            const unsigned int,
            const vec3[float],
            const bool_t) except +
        vector[vec3[float]] getBufferPoints()
        vector[uint] getBufferIds()
