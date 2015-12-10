# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._trajectory as trajectory

cdef extern from "BondOrder.h" namespace "freud::order":
    cdef cppclass BondOrder:
        BondOrder(float, float, unsigned int, unsigned int, unsigned int)
        const trajectory.Box &getBox() const
        void resetBondOrder()
        void accumulate(trajectory.Box &,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int) nogil
        void reduceBondOrder()
        shared_array[float] getBondOrder()
        shared_array[float] getTheta()
        shared_array[float] getPhi()
        unsigned int getNBinsTheta()
        unsigned int getNBinsPhi()

cdef extern from "EntropicBonding.h" namespace "freud::order":
    cdef cppclass EntropicBonding:
        EntropicBonding(float, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *)
        const trajectory.Box &getBox() const
        void compute(trajectory.Box &,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil
        shared_array[unsigned int] getBonds()
        unsigned int getNP()
        unsigned int getNBinsX()
        unsigned int getNBinsY()

# cdef extern from "HexOrderParameter.h" namespace "freud::order":

# cdef extern from "LocalDescriptors.h" namespace "freud::order":

# cdef extern from "TransOrderParameter.h" namespace "freud::order":
