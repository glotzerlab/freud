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

Bootstrap::Bootstrap(const unsigned int nBootstrap, boost::python::numeric::array data_array)
    : m_nBootstrap(nBootstrap)
    {

    num_util::check_type(data_array, NPY_UINT);
    num_util::check_rank(data_array, 1);
    m_arrSize = num_util::shape(data_array)[0];
    unsigned int* data_array_raw = (unsigned int*) num_util::data(data_array);

    m_bootstrap_array = boost::shared_array<unsigned int>(new unsigned int[m_nBootstrap * m_arrSize]);
    memset((void*)m_bootstrap_array.get(), 0, sizeof(unsigned int)*m_nBootstrap * m_arrSize);

    m_avg_array = boost::shared_array<float>(new float[m_arrSize]);
    memset((void*)m_avg_array.get(), 0, sizeof(float)*m_arrSize);

    m_std_array = boost::shared_array<float>(new float[m_arrSize]);
    memset((void*)m_std_array.get(), 0, sizeof(float)*m_arrSize);

    m_err_array = boost::shared_array<float>(new float[m_arrSize]);
    memset((void*)m_err_array.get(), 0, sizeof(float)*m_arrSize);

    m_data_array = new std::vector<unsigned int>(m_arrSize);
    m_cum_array = new std::vector<unsigned int>(m_arrSize);
    // populate the arrays; could be done with a memcpy for m_data_array, but m_cum_array needs the for loop
    (*m_data_array)[0] = (unsigned int) data_array_raw[0];
    (*m_cum_array)[0] = (unsigned int) data_array_raw[0];
    for (unsigned int i = 1; i < m_arrSize; i++)
        {
        (*m_data_array)[i] = (unsigned int) data_array_raw[i];
        (*m_cum_array)[i] = (*m_cum_array)[i-1] + (unsigned int) data_array_raw[i];
        }
    m_nPoints = (*m_cum_array)[m_arrSize-1];
    }

Bootstrap::~Bootstrap()
    {
    }

class ComputeBootstrap
    {
    private:
        atomic<unsigned int> *m_bootstrapArray;
        const std::vector<unsigned int> *m_cum_array;
        const unsigned int m_nBootstrap;
        const unsigned int m_nPoints;
        const unsigned int m_arrSize;
    public:
        ComputeBootstrap(atomic<unsigned int> *bootstrapArray,
                         const std::vector<unsigned int> *cum_array,
                         const unsigned int nBootstrap,
                         const unsigned int nPoints,
                         const unsigned int arrSize)
            : m_bootstrapArray(bootstrapArray), m_cum_array(cum_array), m_nBootstrap(nBootstrap), m_nPoints(nPoints), m_arrSize(arrSize)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {

            Index2D b_i = Index2D(m_arrSize, m_nBootstrap);
            std::vector<unsigned int>::const_iterator iterIDX;
            // for each bootstrap array in the assigned block
            float myCNT = 0;
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                printf("I have %d points to roll\n", m_nPoints);
                for (unsigned int j = 0; j < m_nPoints; j++)
                    {
                    int myRand = (int)(rand() % (int)(m_nPoints));
                    // look up the array index
                    iterIDX = upper_bound((*m_cum_array).begin(), (*m_cum_array).end(), myRand);
                    unsigned int arrIDX = iterIDX - (*m_cum_array).begin();
                    m_bootstrapArray[b_i(arrIDX, i)]++;
                    }
                myCNT += 1;
                // printf("I just finished bootstrap %d\n", (int) i);
                // printf("I am %f done with assigned bootstraps\n", (float)(myCNT / (float) (myR.end() - myR.begin())));
                } // done populating the bootstrap array i
            }
    };

void Bootstrap::AnalyzeBootstrap(boost::shared_array<unsigned int> *bootstrap_array,
                                 boost::shared_array<float> *avg_array,
                                 boost::shared_array<float> *std_array,
                                 boost::shared_array<float> *err_array,
                                 std::vector<unsigned int> *cum_array)
        {
        // calculate the average for each index
        Index2D b_i = Index2D(m_arrSize, m_nBootstrap);
        for (unsigned int i = 0; i < m_arrSize; i++)
            {
            for (unsigned int j = 0; j < m_nBootstrap; j++)
                {
                (*avg_array)[i] += (*bootstrap_array)[j * m_arrSize + i];
                (*avg_array)[i] += (*bootstrap_array)[b_i(i, j)];
                // look up the array index
                }
            (*avg_array)[i] /= m_nBootstrap;
            } // done populating the bootstrap array i
        // calculate the std
        for (unsigned int i = 0; i < m_arrSize; i++)
            {
            float mySTD = 0.0;
            for (unsigned int j = 0; j < m_nBootstrap; j++)
                {
                mySTD += ((*bootstrap_array)[b_i(i, j)] - (*avg_array)[i]) * ((*bootstrap_array)[b_i(i, j)] - (*avg_array)[i]);
                }
            (*std_array)[i] = sqrt((1.0/(float)m_nBootstrap)*mySTD);
            }
        for (unsigned int i = 0; i < m_arrSize; i++)
            {
            (*err_array)[i] = (*avg_array)[i]/(*std_array)[i];
            } // done analyzing the data
        }

void Bootstrap::compute()
    {
    parallel_for(blocked_range<size_t>(0,m_nBootstrap), ComputeBootstrap((atomic<unsigned int>*)m_bootstrap_array.get(),
                                                                         m_cum_array,
                                                                         m_nBootstrap,
                                                                         m_nPoints,
                                                                         m_arrSize));
    AnalyzeBootstrap(&m_bootstrap_array,
                     &m_avg_array,
                     &m_std_array,
                     &m_err_array,
                     m_cum_array);
    }

void Bootstrap::computePy()
    {
    // unlike all other freud functions, this one takes no arguments as the size of the arrays can't change between
    // creation and compute
        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute();
        }
    }

void export_Bootstrap()
    {
    class_<Bootstrap>("Bootstrap", init<unsigned int, boost::python::numeric::array>())
        .def("compute", &Bootstrap::computePy)
        .def("getBootstrap", &Bootstrap::getBootstrapPy)
        .def("getAVG", &Bootstrap::getAVGPy)
        .def("getSTD", &Bootstrap::getSTDPy)
        .def("getERR", &Bootstrap::getERRPy)
        ;
    }

}; }; // end namespace freud::bootstrap
