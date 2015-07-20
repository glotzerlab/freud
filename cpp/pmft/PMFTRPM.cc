#include "PMFTRPM.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include "VectorMath.h"

using namespace std;
using namespace boost::python;

using namespace tbb;

/*! \internal
    \file PMFTRPM.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFTRPM::PMFTRPM(float max_r, float max_TP, float max_TM, unsigned int nbins_r, unsigned int nbins_TP, unsigned int nbins_TM)
    : m_box(trajectory::Box()), m_max_r(max_r), m_max_TP(max_TP), m_max_TM(max_TM),
      m_nbins_r(nbins_r), m_nbins_TP(nbins_TP), m_nbins_TM(nbins_TM)
    {
    if (nbins_r < 1)
        throw invalid_argument("must be at least 1 bin in r");
    if (nbins_TP < 1)
        throw invalid_argument("must be at least 1 bin in TP");
    if (nbins_TM < 1)
        throw invalid_argument("must be at least 1 bin in TM");
    if (max_r < 0.0f)
        throw invalid_argument("max_r must be positive");
    if (max_TP < 0.0f)
        throw invalid_argument("max_TP must be positive");
    if (max_TM < 0.0f)
        throw invalid_argument("max_TM must be positive");
    // calculate dr, dTP, dTM
    m_dr = 2.0 * m_max_r / float(m_nbins_r);
    m_dTP = 2.0 * m_max_TP / float(m_nbins_TP);
    m_dTM = 2.0 * m_max_TM / float(m_nbins_TM);

    if (m_dr > max_r)
        throw invalid_argument("max_r must be greater than dr");
    if (m_dTP > max_TP)
        throw invalid_argument("max_TP must be greater than dTP");
    if (m_dTM > max_TM)
        throw invalid_argument("max_TM must be greater than dTM");

    // precompute the bin center positions for r
    m_r_array = boost::shared_array<float>(new float[m_nbins_r]);
    for (unsigned int i = 0; i < m_nbins_r; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        m_r_array[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
        }

    // precompute the bin center positions for TP
    m_TP_array = boost::shared_array<float>(new float[m_nbins_TP]);
    for (unsigned int i = 0; i < m_nbins_TP; i++)
        {
        float TP = float(i) * m_dTP;
        float nextTP = float(i+1) * m_dTP;
        m_TP_array[i] = -m_max_TP + ((TP + nextTP) / 2.0);
        }

    // precompute the bin center positions for TM
    m_TM_array = boost::shared_array<float>(new float[m_nbins_TM]);
    for (unsigned int i = 0; i < m_nbins_TM; i++)
        {
        float TM = float(i) * m_dTM;
        float nextTM = float(i+1) * m_dTM;
        m_TM_array[i] = -m_max_TM + ((TM + nextTM) / 2.0);
        }

    // create and populate the pcf_array
    m_pcf_array = boost::shared_array<unsigned int>(new unsigned int[m_nbins_r*m_nbins_TP*m_nbins_TM]);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_TP*m_nbins_TM);

    m_lc = new locality::LinkCell(m_box, m_max_r);
    }

PMFTRPM::~PMFTRPM()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_pcf_array.begin(); i != m_local_pcf_array.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_lc;
    }

class CombinePCFRPM
    {
    private:
        unsigned int m_nbins_r;
        unsigned int m_nbins_TP;
        unsigned int m_nbins_TM;
        unsigned int *m_pcf_array;
        tbb::enumerable_thread_specific<unsigned int *>& m_local_pcf_array;
    public:
        CombinePCFRPM(unsigned int nbins_r,
                      unsigned int nbins_TP,
                      unsigned int nbins_TM,
                      unsigned int *pcf_array,
                      tbb::enumerable_thread_specific<unsigned int *>& local_pcf_array)
            : m_nbins_r(nbins_r), m_nbins_TP(nbins_TP), m_nbins_TM(nbins_TM), m_pcf_array(pcf_array),
              m_local_pcf_array(local_pcf_array)
        {
        }
        void operator()( const blocked_range<size_t> &myBin ) const
            {
            Index3D b_i = Index3D(m_nbins_r, m_nbins_TP, m_nbins_TM);
            for (size_t i = myBin.begin(); i != myBin.end(); i++)
                {
                for (size_t j = 0; j < m_nbins_TP; j++)
                    {
                    for (size_t k = 0; k < m_nbins_TP; k++)
                        {
                        for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_pcf_array.begin();
                             local_bins != m_local_pcf_array.end(); ++local_bins)
                            {
                            m_pcf_array[b_i((int)i, (int)j, (int)k)] += (*local_bins)[b_i((int)i, (int)j, (int)k)];
                            }
                        }
                    }
                }
            }
    };

class ComputePMFTRPM
    {
    private:
        tbb::enumerable_thread_specific<unsigned int *>& m_pcf_array;
        unsigned int m_nbins_r;
        unsigned int m_nbins_TP;
        unsigned int m_nbins_TM;
        const trajectory::Box m_box;
        const float m_max_r;
        const float m_max_TP;
        const float m_max_TM;
        const float m_dr;
        const float m_dTP;
        const float m_dTM;
        const locality::LinkCell *m_lc;
        vec3<float> *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        vec3<float> *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFTRPM(tbb::enumerable_thread_specific<unsigned int *>& pcf_array,
                       unsigned int nbins_r,
                       unsigned int nbins_TP,
                       unsigned int nbins_TM,
                       const trajectory::Box &box,
                       const float max_r,
                       const float max_TP,
                       const float max_TM,
                       const float dr,
                       const float dTP,
                       const float dTM,
                       const locality::LinkCell *lc,
                       vec3<float> *ref_points,
                       float *ref_orientations,
                       unsigned int Nref,
                       vec3<float> *points,
                       float *orientations,
                       unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_r(nbins_r), m_nbins_TP(nbins_TP), m_nbins_TM(nbins_TM), m_box(box),
              m_max_r(max_r), m_max_TP(max_TP), m_max_TM(max_TM), m_dr(dr), m_dTP(dTP), m_dTM(dTM), m_lc(lc),
              m_ref_points(ref_points), m_ref_orientations(ref_orientations), m_Nref(Nref), m_points(points),
              m_orientations(orientations), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_points);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            float dr_inv = 1.0f / m_dr;
            float maxrsq = m_max_r * m_max_r;
            float dTP_inv = 1.0f / m_dTP;
            float dTM_inv = 1.0f / m_dTM;

            Index3D b_i = Index3D(m_nbins_r, m_nbins_TP, m_nbins_TM);

            bool exists;
            m_pcf_array.local(exists);
            if (! exists)
                {
                m_pcf_array.local() = new unsigned int [m_nbins_r*m_nbins_TP*m_nbins_TM];
                memset((void*)m_pcf_array.local(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_TP*m_nbins_TM);
                }

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                // get the cell the point is in
                vec3<float> ref = m_ref_points[i];
                unsigned int ref_cell = m_lc->getCell(ref);

                // loop over all neighboring cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    unsigned int neigh_cell = neigh_cells[neigh_idx];

                    // iterate over the particles in that cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                        {
                        vec3<float> delta = m_box.wrap(m_points[j] - ref);

                        float rsq = dot(delta, delta);
                        if (rsq < 1e-6)
                            {
                            continue;
                            }
                        if (rsq < maxrsq)
                            {
                            float r = sqrtf(rsq);
                            // calculate angles
                            float dTheta1 = atan2(delta.y, delta.x);
                            float dTheta2 = atan2(-delta.y, -delta.x);
                            float T1 = dTheta1 - m_ref_orientations[i];
                            float T2 = dTheta2 - m_orientations[j];
                            float TP = T1 + T2 + m_max_TP;
                            float TM = T1 - T2 + m_max_TM;

                            // bin that point
                            float binr = r * dr_inv;
                            float binTP = floorf(TP * dTP_inv);
                            float binTM = floorf(TM * dTM_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinr = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                            unsigned int ibinTP = _mm_cvtt_ss2si(_mm_load_ss(&binTP));
                            unsigned int ibinTM = _mm_cvtt_ss2si(_mm_load_ss(&binTM));
                            #else
                            unsigned int ibinr = (unsigned int)(binr);
                            unsigned int ibinTP = (unsigned int)(binTP);
                            unsigned int ibinTM = (unsigned int)(binTM);
                            #endif

                            if ((ibinr < m_nbins_r) && (ibinTP < m_nbins_TP) && (ibinTM < m_nbins_TM))
                                {
                                ++m_pcf_array.local()[b_i(ibinr, ibinTP, ibinTM)];
                                }
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

//! \internal
//! helper function to reduce the thread specific arrays into the boost array
void PMFTRPM::reducePCF()
    {
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_TP*m_nbins_TM);
    parallel_for(blocked_range<size_t>(0,m_nbins_r),
                 CombinePCFRPM(m_nbins_r,
                               m_nbins_TP,
                               m_nbins_TM,
                               m_pcf_array.get(),
                               m_local_pcf_array));
    }

//! Get a reference to the PCF array
boost::shared_array<unsigned int> PMFTRPM::getPCF()
    {
    reducePCF();
    return m_pcf_array;
    }

//! Get a reference to the PCF array
boost::python::numeric::array PMFTRPM::getPCFPy()
    {
    reducePCF();
    unsigned int *arr = m_pcf_array.get();
    std::vector<intp> dims(3);
    dims[0] = m_nbins_TM;
    dims[1] = m_nbins_TP;
    dims[2] = m_nbins_r;
    return num_util::makeNum(arr, dims);
    }

void PMFTRPM::resetPCF()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_pcf_array.begin(); i != m_local_pcf_array.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_TP*m_nbins_TM);
        }
    // memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_TP*m_nbins_TM);
    }

void PMFTRPM::accumulate(vec3<float> *ref_points,
                         float *ref_orientations,
                         unsigned int Nref,
                         vec3<float> *points,
                         float *orientations,
                         unsigned int Np)
    {
    m_lc->computeCellList(m_box, points, Np);
    parallel_for(blocked_range<size_t>(0,Nref),
                 ComputePMFTRPM(m_local_pcf_array,
                                m_nbins_r,
                                m_nbins_TP,
                                m_nbins_TM,
                                m_box,
                                m_max_r,
                                m_max_TP,
                                m_max_TM,
                                m_dr,
                                m_dTP,
                                m_dTM,
                                m_lc,
                                ref_points,
                                ref_orientations,
                                Nref,
                                points,
                                orientations,
                                Np));
    }

//! \internal
/*! \brief Exposed function to python to calculate the PMF
*/
void PMFTRPM::accumulatePy(trajectory::Box& box,
                           boost::python::numeric::array ref_points,
                           boost::python::numeric::array ref_orientations,
                           boost::python::numeric::array points,
                           boost::python::numeric::array orientations)
    {
    // validate input type and rank
    m_box = box;
    num_util::check_type(ref_points, NPY_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(ref_orientations, NPY_FLOAT);
    num_util::check_rank(ref_orientations, 1);
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(orientations, NPY_FLOAT);
    num_util::check_rank(orientations, 1);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];

    // check the size of angles to be correct
    num_util::check_dim(ref_orientations, 0, Nref);
    num_util::check_dim(orientations, 0, Np);

    // get the raw data pointers and compute the cell list
    vec3<float>* ref_points_raw = (vec3<float>*) num_util::data(ref_points);
    float* ref_orientations_raw = (float*) num_util::data(ref_orientations);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    float* orientations_raw = (float*) num_util::data(orientations);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        accumulate(ref_points_raw,
                ref_orientations_raw,
                Nref,
                points_raw,
                orientations_raw,
                Np);
        }
    }

//! \internal
/*! \brief Exposed function to python to calculate the PMF
*/
void PMFTRPM::computePy(trajectory::Box& box,
                        boost::python::numeric::array ref_points,
                        boost::python::numeric::array ref_orientations,
                        boost::python::numeric::array points,
                        boost::python::numeric::array orientations)
    {
    resetPCF();
    accumulatePy(box, ref_points, ref_orientations, points, orientations);
    }

void export_PMFTRPM()
    {
    class_<PMFTRPM>("PMFTRPM", init<float, float, float, float, float, float>())
        .def("getBox", &PMFTRPM::getBox, return_internal_reference<>())
        .def("accumulate", &PMFTRPM::accumulatePy)
        .def("compute", &PMFTRPM::computePy)
        .def("getPCF", &PMFTRPM::getPCFPy)
        .def("resetPCF", &PMFTRPM::resetPCFPy)
        .def("getR", &PMFTRPM::getRPy)
        .def("getTP", &PMFTRPM::getTPPy)
        .def("getTM", &PMFTRPM::getTMPy)
        ;
    }

}; }; // end namespace freud::pmft
