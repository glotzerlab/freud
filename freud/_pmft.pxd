# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.memory cimport shared_ptr
cimport freud._box as box

cdef extern from "PMFTR12.h" namespace "freud::pmft":
    cdef cppclass PMFTR12:
        PMFTR12(float, unsigned int, unsigned int, unsigned int)

        const box.Box& getBox() const
        void resetPCF()
        void accumulate(box.Box&,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        void reducePCF()
        shared_ptr[unsigned int] getBinCounts()
        shared_ptr[float] getPCF()
        shared_ptr[float] getR()
        shared_ptr[float] getT1()
        shared_ptr[float] getT2()
        shared_ptr[float] getInverseJacobian()
        unsigned int getNBinsR()
        unsigned int getNBinsT1()
        unsigned int getNBinsT2()
        float getRCut()

cdef extern from "PMFTXYT.h" namespace "freud::pmft":
    cdef cppclass PMFTXYT:
        PMFTXYT(float, float, unsigned int, unsigned int, unsigned int)

        const box.Box& getBox() const
        void resetPCF()
        void accumulate(box.Box&,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        void reducePCF()
        shared_ptr[unsigned int] getBinCounts()
        shared_ptr[float] getPCF()
        shared_ptr[float] getX()
        shared_ptr[float] getY()
        shared_ptr[float] getT()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsT()
        float getRCut()

cdef extern from "PMFTXY2D.h" namespace "freud::pmft":
    cdef cppclass PMFTXY2D:
        PMFTXY2D(float, unsigned int, unsigned int, unsigned int)

        const box.Box& getBox() const
        void resetPCF()
        void accumulate(box.Box&,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        void reducePCF()
        shared_ptr[unsigned int] getBinCounts()
        shared_ptr[float] getPCF()
        shared_ptr[float] getX()
        shared_ptr[float] getY()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        float getRCut()

cdef extern from "PMFTXYZ.h" namespace "freud::pmft":
    cdef cppclass PMFTXYZ:
        PMFTXYZ(float, float, float, unsigned int, unsigned int, unsigned int)

        const box.Box& getBox() const
        void resetPCF()
        void accumulate(box.Box&,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        quat[float]*,
                        unsigned int) nogil
        void reducePCF()
        shared_ptr[float] getPCF()
        shared_ptr[unsigned int] getBinCounts()
        shared_ptr[float] getX()
        shared_ptr[float] getY()
        shared_ptr[float] getZ()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsZ()
        float getRCut()

cdef extern from "PMFTRtheta.h" namespace "freud::pmft":
    cdef cppclass PMFTRtheta:
        PMFTRtheta(float, float, unsigned int, unsigned int)

        const box.Box& getBox() const
        void resetPCF()
        void accumulate(box.Box&,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        quat[float]*,
                        unsigned int) nogil
        void reducePCF()
        shared_ptr[float] getPCF()
        shared_ptr[unsigned int] getBinCounts()
        shared_ptr[float] getR()
        shared_ptr[float] get_theta()
        float getJacobian()
        unsigned int getNBinsR()
        unsigned int getNBins_theta()
        float getRCut()
