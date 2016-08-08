from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
cimport freud._box as box

cdef extern from "BondingAnalysis.h" namespace "freud::bond":
    cdef cppclass BondingAnalysis:
        BondingAnalysis(unsigned int, unsigned int)
        void reduceArrays()
        void compute(unsigned int*, unsigned int*) nogil
        vector[vector[uint]] getBondLifetimes()
        vector[vector[uint]] getOverallLifetimes()
        shared_ptr[uint] getTransitionMatrix()
        unsigned int getNumFrames()
        unsigned int getNumParticles()
        unsigned int getNumBonds()

cdef extern from "BondingR12.h" namespace "freud::bond":
    cdef cppclass BondingR12:
        BondingR12(float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *, unsigned int *)
        const box.Box &getBox() const
        void compute(box.Box &,
                     vec3[float]*,
                     float*,
                     unsigned int,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil
        shared_ptr[ uint ] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[ uint, uint] getListMap()
        map[ uint, uint] getRevListMap()

cdef extern from "BondingXY2D.h" namespace "freud::bond":
    cdef cppclass BondingXY2D:
        BondingXY2D(float, float, unsigned int, unsigned int, unsigned int, unsigned int *, unsigned int *)
        const box.Box &getBox() const
        void compute(box.Box &,
                     vec3[float]*,
                     float*,
                     unsigned int,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil
        shared_ptr[ uint ] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[ uint, uint] getListMap()
        map[ uint, uint] getRevListMap()

cdef extern from "BondingXYT.h" namespace "freud::bond":
    cdef cppclass BondingXYT:
        BondingXYT(float, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *, unsigned int *)
        const box.Box &getBox() const
        void compute(box.Box &,
                     vec3[float]*,
                     float*,
                     unsigned int,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil
        shared_ptr[ uint ] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[ uint, uint] getListMap()
        map[ uint, uint] getRevListMap()

cdef extern from "BondingXYZ.h" namespace "freud::bond":
    cdef cppclass BondingXYZ:
        BondingXYZ(float, float, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *,
            unsigned int *)
        const box.Box &getBox() const
        void compute(box.Box &,
                     vec3[float]*,
                     quat[float]*,
                     unsigned int,
                     vec3[float]*,
                     quat[float]*,
                     unsigned int) nogil
        shared_ptr[ uint ] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[ uint, uint] getListMap()
        map[ uint, uint] getRevListMap()
