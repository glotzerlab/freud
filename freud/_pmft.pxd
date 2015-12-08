# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._trajectory as trajectory

cdef extern from "PMFTR12.h" namespace "freud::pmft":
    cdef cppclass PMFTR12:
        PMFTR12(float, unsigned int, unsigned int, unsigned int)

        const trajectory.Box& getBox() const
        void resetPCF()
        void accumulate(trajectory.Box&,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        void reducePCF()
        shared_array[unsigned int] getPCF()
        shared_array[float] getR()
        shared_array[float] getT1()
        shared_array[float] getT2()
        unsigned int getNBinsR()
        unsigned int getNBinsT1()
        unsigned int getNBinsT2()

cdef extern from "PMFXY2D.h" namespace "freud::pmft":
    cdef cppclass PMFXY2D:
        PMFXY2D(float, unsigned int, unsigned int, unsigned int)

        const trajectory.Box& getBox() const
        void resetPCF()
        void accumulate(trajectory.Box&,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        void reducePCF()
        shared_array[unsigned int] getPCF()
        shared_array[float] getX()
        shared_array[float] getY()
        unsigned int getNBinsX()
        unsigned int getNBinsY()

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
