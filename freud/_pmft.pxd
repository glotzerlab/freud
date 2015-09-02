# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._trajectory as trajectory

cdef extern from "PMFXYZ.h" namespace "freud::pmft":
    cdef cppclass PMFXYZ:
        PMFXYZ(float, float, float, unsigned int, unsigned int, unsigned int)

        const trajectory.Box& getBox() const
        void resetPCF()
        void accumulate(trajectory.Box&,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        quat[float]*,
                        unsigned int) nogil
        void reducePCF()
        shared_array[unsigned int] getPCF()
        shared_array[float] getX()
        shared_array[float] getY()
        shared_array[float] getZ()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsZ()
