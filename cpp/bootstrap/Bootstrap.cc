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

Bootstrap::Bootstrap(const unsigned int nBootstrap, const unsigned int nPoints, const unsigned int arrSize)
    : m_nBootstrap(nBootstrap), m_nPoints(nPoints), m_arrSize(arrSize)
    {
    }

Bootstrap::~Bootstrap()
    {
    }

int Bootstrap::compareInts(const void * a, const void * b)
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

class ComputeBootstrap
    {
    private:
        atomic<unsigned int> *m_bootstrapArray;
        atomic<unsigned int> *m_dataCum;
    public:
        ComputeBootstrap(atomic<unsigned int> *bootstrapArray,
                         atomic<unsigned int> *dataCum)
            : m_bootstrapArray(bootstrapArray), m_dataCum(dataCum)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {

            // for each bootstrap array in the assigned block
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                for (unsigned int j = 0; j < m_nPoints; j++)
                    {
                    int myRand = (int)(rand() % (int)(m_nPoints + 1));
                    // look up the array index
                    int arrIDX = (int*) bsearch(&myRand, m_dataCum, m_arrSize, sizeof(int), compareInts);
                    m_bootstrapArray[i][arrIDX]++;
                    }
                } // done populating the bootstrap array i
            }
    };

void Bootstrap::AnalyzeBootstrap(unsigned int *bootstrapArray,
                                 unsigned int *bootstrapAVG,
                                 unsigned int *bootstrapSTD,
                                 unsigned int *bootstrapRatio,
                                 unsigned int *dataCum)
        {
        // calculate the average for each index
        for (unsigned int i = 0; i <= m_arrSize; i++)
            {
            for (unsigned int j = 0; j < m_nBootstrap; j++)
                {
                bootstrapAVG[i] += bootstrapArray[j][i];
                // look up the array index
                }
            bootstrapAVG[i] /= m_nBootstrap;
            } // done populating the bootstrap array i
        }
        // calculate the std and ratio for each index
        for (unsigned int i = 0; i <= m_arrSize; i++)
            {
            float mySTD = 0;
            for (unsigned int j = 0; j < m_nBootstrap; j++)
                {
                mySTD += (bootstrapArray[j][i] - bootstrapAVG[i]) * (bootstrapArray[j][i] - bootstrapAVG[i]);
                // look up the array index
                }
            bootstrapSTD[i] = sqrt(1/m_nBootstrap*mySTD);
            bootstrapRatio[i] = dataCum[i] / bootstrapSTD[i];
        } // done analyzing the data
        }
    };

void Bootstrap::compute(unsigned int *bootstrapArray,
                        float *bootstrapAVG,
                        float *bootstrapSTD,
                        float *bootstrapRatio,
                        unsigned int *dataCum)
    {
    parallel_for(blocked_range<size_t>(0,m_nBootstrap), ComputeBootstrap((atomic<unsigned int>*)bootstrapArray, (atomic<unsigned int>*)dataCum));
    AnalyzeBootstrap(bootstrapArray,
                     bootstrapAVG,
                     bootstrapSTD,
                     bootstrapRatio,
                     dataCum)
    }

void Bootstrap::computePy(boost::python::numeric::array bootstrapArray,
                          boost::python::numeric::array bootstrapAVG,
                          boost::python::numeric::array bootstrapSTD,
                          boost::python::numeric::array bootstrapRatio,
                          boost::python::numeric::array dataCum)
    {
    // validate input type and rank
    num_util::check_type(bootstrapArray, PyArray_INT);
    num_util::check_rank(bootstrapArray, 2);
    num_util::check_type(bootstrapAVG, PyArray_FLOAT);
    num_util::check_rank(bootstrapAVG, 1);
    num_util::check_type(bootstrapSTD, PyArray_FLOAT);
    num_util::check_rank(bootstrapSTD, 1);
    num_util::check_type(bootstrapRatio, PyArray_FLOAT);
    num_util::check_rank(bootstrapRatio, 1);
    num_util::check_type(dataCum, PyArray_INT);
    num_util::check_rank(dataCum, 1);

    // validate array dims
    num_util::check_dim(bootstrapArray, 0, m_nBootstrap);
    num_util::check_dim(bootstrapArray, 1, m_arrSize);
    num_util::check_dim(bootstrapAVG, 0, m_arrSize);
    num_util::check_dim(bootstrapSTD, 0, m_arrSize);
    num_util::check_dim(bootstrapRatio, 0, m_arrSize);
    num_util::check_dim(dataCum, 0, m_arrSize);

    // get the raw data pointers and compute the cell list
    unsigned int* bootstrapArray_raw = (unsigned int*) num_util::data(bootstrapArray);
    float* bootstrapAVG_raw = (float*) num_util::data(bootstrapAVG);
    float* bootstrapSTD_raw = (float*) num_util::data(bootstrapSTD);
    float* bootstrapRatio_raw = (float*) num_util::data(bootstrapRatio);
    unsigned int* dataCum_raw = (unsigned int*) num_util::data(dataCum);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(bootstrapArray_raw,
                bootstrapAVG_raw,
                bootstrapSTD_raw,
                bootstrapRatio_raw,
                dataCum_raw);
        }
    }

void export_Bootstrap()
    {
    class_<Bootstrap>("Bootstrap", unsigned int, unsigned int, unsigned int>())
        .def("compute", &Bootstrap::computePy)
        ;
    }

}; }; // end namespace freud::bootstrap
