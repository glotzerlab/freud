from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
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

cdef extern from "CubaticOrderParameter.h" namespace "freud::order":
    cdef cppclass CubaticOrderParameter:
        CubaticOrderParameter(float, float, float, float*, unsigned int, unsigned int)
        void resetCubaticOrderParameter()
        void compute(quat[float]*,
                     unsigned int,
                     unsigned int) nogil
        void reduceCubaticOrderParameter()
        unsigned int getNumParticles()
        float getCubaticOrderParameter()
        shared_ptr[float] getParticleCubaticOrderParameter()
        shared_ptr[float] getParticleTensor()
        shared_ptr[float] getGlobalTensor()
        shared_ptr[float] getCubaticTensor()
        shared_ptr[float] getGenR4Tensor()
        float getTInitial()
        float getTFinal()
        float getScale()
        quat[float] getCubaticOrientation()

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

cdef extern from "LocalQl.h" namespace "freud::order":
    cdef cppclass LocalQl:
        LocalQl(const trajectory.Box&, float, unsigned int, float)
        const trajectory.Box& getBox() const
        void setBox(const trajectory.Box)
        void compute(const vec3[float]*,
                     unsigned int)
        void computeAve(const vec3[float]*,
                        unsigned int)
        void computeNorm(const vec3[float]*,
                         unsigned int)
        void computeAveNorm(const vec3[float]*,
                            unsigned int)
        shared_array[float] getQl()
        shared_array[float] getAveQl()
        shared_array[float] getQlNorm()
        shared_array[float] getQlAveNorm()
        unsigned int getNP()


cdef extern from "LocalQlNear.h" namespace "freud::order":
    cdef cppclass LocalQlNear:
        LocalQlNear(const trajectory.Box&, float, unsigned int, unsigned int)
        const trajectory.Box& getBox() const
        void setBox(const trajectory.Box)
        void compute(const vec3[float]*,
                     unsigned int)
        void computeAve(const vec3[float]*,
                        unsigned int)
        void computeNorm(const vec3[float]*,
                         unsigned int)
        void computeAveNorm(const vec3[float]*,
                            unsigned int)
        shared_array[float] getQl()
        shared_array[float] getAveQl()
        shared_array[float] getQlNorm()
        shared_array[float] getQlAveNorm()
        unsigned int getNP()

cdef extern from "LocalWl.h" namespace "freud::order":
    cdef cppclass LocalWl:
        LocalWl(const trajectory.Box&, float, unsigned int)
        const trajectory.Box& getBox() const
        void setBox(const trajectory.Box)
        void compute(const vec3[float]*,
                     unsigned int)
        void computeAve(const vec3[float]*,
                        unsigned int)
        void computeNorm(const vec3[float]*,
                         unsigned int)
        void computeAveNorm(const vec3[float]*,
                            unsigned int)
        shared_array[float] getQl()
        shared_array[float complex] getWl()
        shared_array[float complex] getAveWl()
        shared_array[float complex] getWlNorm()
        shared_array[float complex] getAveNormWl()
        void enableNormalization()
        void disableNormalization()
        unsigned int getNP()

cdef extern from "LocalWlNear.h" namespace "freud::order":
    cdef cppclass LocalWlNear:
        LocalWlNear(const trajectory.Box&, float, unsigned int, unsigned int)
        const trajectory.Box& getBox() const
        void setBox(const trajectory.Box)
        void compute(const vec3[float]*,
                     unsigned int)
        void computeAve(const vec3[float]*,
                        unsigned int)
        void computeNorm(const vec3[float]*,
                         unsigned int)
        void computeAveNorm(const vec3[float]*,
                            unsigned int)
        shared_array[float] getQl()
        shared_array[float complex] getWl()
        shared_array[float complex] getWlNorm()
        shared_array[float complex] getAveWl()
        shared_array[float complex] getWlAveNorm()
        void enableNormalization()
        void disableNormalization()
        unsigned int getNP()

cdef extern from "SolLiq.h" namespace "freud::order":
    cdef cppclass SolLiq:
        SolLiq(const trajectory.Box&, float, float, unsigned int, unsigned int)
        const trajectory.Box& getBox() const
        void setBox(const trajectory.Box)
        void setClusteringRadius(float)
        void compute(const vec3[float]*,
                     unsigned int)
        void computeSolLiqVariant(const vec3[float]*,
                                  unsigned int)
        void computeSolLiqNoNorm(const vec3[float]*,
                                 unsigned int)
        unsigned int getLargestClusterSize()
        vector[unsigned int] getClusterSizes()
        shared_array[float complex] getQlmi()
        shared_array[unsigned int] getClusters()
        shared_array[unsigned int] getNumberOfConnections()
        vector[float complex] getQldot_ij()
        unsigned int getNP()
        unsigned int getNumClusters()

cdef extern from "MatchEnv.h" namespace "freud::order":
    cdef cppclass MatchEnv:
        MatchEnv(const trajectory.Box&, float, unsigned int)
        void setBox(const trajectory.Box)
        void cluster(const vec3[float]*,
                     unsigned int,
                     float,
                     bool,
                     bool,
                     bool)
        void matchMotif(const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int,
                        float,
                        bool)
        map[unsigned int, unsigned int] isSimilar(const vec3[float]*,
                                        vec3[float]*,
                                        unsigned int,
                                        float,
                                        bool)
        map[unsigned int, unsigned int] minimizeRMSD(const vec3[float]*,
                                        vec3[float]*,
                                        unsigned int,
                                        float&,
                                        bool)
        shared_array[unsigned int] getClusters()
        shared_array[vec3[float]] getEnvironment(unsigned int)
        shared_array[vec3[float]] getTotEnvironment()
        unsigned int getNP()
        unsigned int getNumClusters()
        unsigned int getNumNeighbors()
        unsigned int getMaxNumNeighbors()

cdef extern from "SolLiqNear.h" namespace "freud::order":
    cdef cppclass SolLiqNear:
        SolLiqNear(const trajectory.Box&, float, float, unsigned int, unsigned int, unsigned int)
        const trajectory.Box& getBox() const
        void setBox(const trajectory.Box)
        void setClusteringRadius(float)
        void compute(const vec3[float]*,
                     unsigned int)
        void computeSolLiqVariant(const vec3[float]*,
                                  unsigned int)
        void computeSolLiqNoNorm(const vec3[float]*,
                                 unsigned int)
        unsigned int getLargestClusterSize()
        vector[unsigned int] getClusterSizes()
        shared_array[float complex] getQlmi()
        shared_array[unsigned int] getClusters()
        shared_array[unsigned int] getNumberOfConnections()
        vector[float complex] getQldot_ij()
        unsigned int getNumClusters()
        unsigned int getNP()

cdef extern from "Pairing2D.h" namespace "freud::order":
    cdef cppclass Pairing2D:
        Pairing2D(const float, const unsigned int, float)
        const trajectory.Box &getBox() const
        void resetBondOrder()
        void compute(trajectory.Box &,
                     vec3[float]*,
                     float*,
                     float*,
                     unsigned int,
                     unsigned int)
        shared_array[unsigned int] getMatch()
        shared_array[unsigned int] getPair()
        unsigned int getNumParticles()
