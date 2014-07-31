#include "Bootstrap.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

#include "VectorMath.h"

using namespace std;
using namespace boost::python;

using namespace tbb;

/*! \file Bootstrap.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace bootstrap {

inline int compareInts(const void * a, const void * b)
    {
    if ( *(int*)a <=  *(int*)b )
        {
        return 0;
        }
    else
        {
        return 1;
        }
    }

Bootstrap::Bootstrap(const unsigned int nBootstrap, const unsigned int nPoints, const unsigned int arrSize)
    : m_nBootstrap(nBootstrap), m_nPoints(nPoints), m_arrSize(arrSize)
    {
    }

Bootstrap::~Bootstrap()
    {
    }

class ComputeBootstrap
    {
    private:
        atomic<unsigned int> *m_bootstrapArray;
        const std::vector<unsigned int> m_dataCum;
        const unsigned int m_nBootstrap;
        const unsigned int m_nPoints;
        const unsigned int m_arrSize;
    public:
        ComputeBootstrap(atomic<unsigned int> *bootstrapArray,
                         const std::vector<unsigned int> dataCum,
                         const unsigned int nBootstrap,
                         const unsigned int nPoints,
                         const unsigned int arrSize)
            : m_bootstrapArray(bootstrapArray), m_dataCum(dataCum), m_nBootstrap(nBootstrap), m_nPoints(nPoints), m_arrSize(arrSize)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {


            std::vector<unsigned int>::const_iterator iterIDX;
            // for each bootstrap array in the assigned block
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                for (unsigned int j = 0; j < m_nPoints; j++)
                    {
                    // roll the random integer to sample the array
                    int myRand = (int)(rand() % (int)(m_nPoints));
                    // find the bin to iterate from the array
                    iterIDX = upper_bound(m_dataCum.begin(), m_dataCum.end(), myRand);
                    unsigned int arrIDX = iterIDX - m_dataCum.begin();
                    // iterate that bin
                    m_bootstrapArray[i * m_arrSize + arrIDX]++;
                    }
                } // done populating the bootstrap array i
            }
    };

void Bootstrap::AnalyzeBootstrap(std::vector<unsigned int> bootstrapArray,
                                 float *bootstrapAVG,
                                 float *bootstrapSTD,
                                 float *bootstrapRatio,
                                 unsigned int *dataCum,
                                 unsigned int *dataFlat)
        {
        // calculate the average for each index
        for (unsigned int i = 0; i < m_arrSize; i++)
            {
            for (unsigned int j = 0; j < m_nBootstrap; j++)
                {
                bootstrapAVG[i] += (float)bootstrapArray[j * m_arrSize + i];
                // look up the array index
                }
            bootstrapAVG[i] /= float(m_nBootstrap);
            } // done populating the bootstrap array i
        // calculate the std and ratio for each index
        for (unsigned int i = 0; i < m_arrSize; i++)
            {
            float mySTD = 0.0;
            for (unsigned int j = 0; j < m_nBootstrap; j++)
                {
                mySTD += ((float)bootstrapArray[j * m_arrSize + i] - bootstrapAVG[i]) * ((float)bootstrapArray[j * m_arrSize + i] - bootstrapAVG[i]);
                // pass in the true mean
                bootstrapRatio[i] += (abs(float((float)dataFlat[i] - (float)bootstrapArray[j * m_arrSize + i])) / ((float)dataFlat[i]));
                // look up the array index
                }
            bootstrapSTD[i] = sqrt((1.0/(float)m_nBootstrap)*mySTD);
            bootstrapRatio[i] /= float(m_nBootstrap);
            } // done analyzing the data
        }

void Bootstrap::compute(float *bootstrapAVG,
                        float *bootstrapSTD,
                        float *bootstrapRatio,
                        unsigned int *dataCum,
                        unsigned int *dataFlat)
    {
    std::vector<unsigned int> dataCumCopy (m_arrSize);
    std::vector<unsigned int> bootstrapArray (m_nBootstrap * m_arrSize);
    for (unsigned int i = 0; i < m_arrSize; i++)
        {
        dataCumCopy[i] = dataCum[i];
        for (unsigned int j = 0; j < m_nBootstrap; j++)
            {
            bootstrapArray[j * m_arrSize + i] = 0;
            }
        }
    parallel_for(blocked_range<size_t>(0,m_nBootstrap), ComputeBootstrap((atomic<unsigned int>*) &bootstrapArray, dataCumCopy, m_nBootstrap, m_nPoints, m_arrSize));
    AnalyzeBootstrap(bootstrapArray,
                     bootstrapAVG,
                     bootstrapSTD,
                     bootstrapRatio,
                     dataCum,
                     dataFlat);
    }

void Bootstrap::computePy(boost::python::numeric::array bootstrapAVG,
                          boost::python::numeric::array bootstrapSTD,
                          boost::python::numeric::array bootstrapRatio,
                          boost::python::numeric::array dataCum,
                          boost::python::numeric::array dataFlat)
    {
    // validate input type and rank
    // these are not correct and need to be changed
    num_util::check_type(dataCum, PyArray_UINT);
    num_util::check_rank(dataCum, 1);
    num_util::check_type(dataFlat, PyArray_UINT);
    num_util::check_rank(dataFlat, 1);

    // validate array dims
    num_util::check_dim(dataCum, 0, m_arrSize);
    num_util::check_dim(dataFlat, 0, m_arrSize);

    // get the raw data pointers and compute the cell list
    float* bootstrapAVG_raw = (float*) num_util::data(bootstrapAVG);
    float* bootstrapSTD_raw = (float*) num_util::data(bootstrapSTD);
    float* bootstrapRatio_raw = (float*) num_util::data(bootstrapRatio);
    unsigned int* dataCum_raw = (unsigned int*) num_util::data(dataCum);
    unsigned int* dataFlat_raw = (unsigned int*) num_util::data(dataFlat);
    // printf("I have the pointers\n");

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(bootstrapAVG_raw,
                bootstrapSTD_raw,
                bootstrapRatio_raw,
                dataCum_raw,
                dataFlat_raw);
        }
    }

void export_Bootstrap()
    {
    class_<Bootstrap>("Bootstrap", init<unsigned int, unsigned int, unsigned int>())
        .def("compute", &Bootstrap::computePy)
        ;
    }

}; }; // end namespace freud::bootstrap
