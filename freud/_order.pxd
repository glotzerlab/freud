# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
from libcpp.complex cimport complex
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

cdef extern from "HexOrderParameter.h" namespace "freud::order":
    cdef cppclass HexOrderParameter:
        HexOrderParameter(float, float, unsigned int)
        const trajectory.Box &getBox() const
        void compute(trajectory.Box &,
                     const vec3[float]*,
                     unsigned int) nogil
        # unsure how to pass back the std::complex, but this seems to compile...
        shared_array[float complex] getPsi()
        unsigned int getNP()
        float getK()

cdef extern from "LocalDescriptors.h" namespace "freud::order":
    cdef cppclass LocalDescriptors:
        LocalDescriptors(const trajectory.Box &,
                         unsigned int,
                         unsigned int,
                         float)
        const trajectory.Box &getBox() const
        unsigned int getNNeigh() const
        unsigned int getLMax() const
        float getRMax() const
        unsigned int getNP()
        void compute(const vec3[float]*,
                     const quat[float]*,
                     unsigned int) nogil
        shared_array[float] getMagR()
        shared_array[quat[float]] getQij()
        shared_array[float complex] getSph()

cdef extern from "TransOrderParameter.h" namespace "freud::order":
    cdef cppclass TransOrderParameter:
        TransOrderParameter(float, float, unsigned int)
        const trajectory.Box &getBox() const,
        void compute(trajectory.Box &,
                     const vec3[float]*,
                     unsigned int) nogil
        shared_array[float complex] getDr()
        unsigned int getNP()
