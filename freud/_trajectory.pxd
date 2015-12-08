from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from libcpp.string cimport string
from freud.util._Boost cimport shared_array

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
        vec3[float] makeCoordinates(const vec3[float]&) const
        vec3[float] makeFraction(const vec3[float]&) const
        vec3[float] getLatticeVector(unsigned int i) const
        vec3[float] wrap(vec3[float]&)

cdef extern from "DCDLoader.h" namespace "freud::trajectory":
    cdef cppclass DCDLoader:
        DCDLoader(const string &)

        void jumpToFrame(int)
        void readNextFrame()

        const Box &getBox() const
        unsigned int getNumParticles() const
        unsigned int getLastFrameNum() const
        unsigned int getFrameCount() const
        string getFileName() const
        unsigned int getTimeStep() const

        shared_array[float] getPoints() const
