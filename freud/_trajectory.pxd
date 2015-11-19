from libcpp cimport bool
from freud.util._VectorMath cimport vec3

cdef extern from "trajectory.h" namespace "freud::trajectory":
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
        vec3[float] makeCoordinates(const vec3[float]&)
        vec3[float] wrap(vec3[float]&)
