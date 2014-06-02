#include "PMFTXYT2D.h"
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

/*! \file PMFTXYT2D.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFTXYT2D::PMFTXYT2D(const trajectory::Box& box, float max_x, float max_y, float max_T, float dx, float dy, float dT)
    : m_box(box), m_max_x(max_x), m_max_y(max_y), m_max_T(max_T), m_dx(dx), m_dy(dy), m_dT(dT)
    {
    if (dx < 0.0f)
        throw invalid_argument("dx must be positive");
    if (dy < 0.0f)
        throw invalid_argument("dy must be positive");
    if (dT < 0.0f)
        throw invalid_argument("dT must be positive");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    if (max_T < 0.0f)
        throw invalid_argument("max_T must be positive");
    if (dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (dy > max_y)
        throw invalid_argument("max_y must be greater than dy");
    if (dT > max_T)
        throw invalid_argument("max_T must be greater than dT");
    if (max_x > box.getLx()/2 || max_y > box.getLy()/2)
        throw invalid_argument("max_x, max_y must be smaller than half the smallest box size");
    if (!box.is2D())
        throw invalid_argument("box must be 2D");

    m_nbins_x = int(floorf(2 * m_max_x / m_dx));
    assert(m_nbins_x > 0);
    m_nbins_y = int(floorf(2 * m_max_y / m_dy));
    assert(m_nbins_y > 0);
    m_nbins_T = int(floorf(2 * m_max_T / m_dT));
    assert(m_nbins_T > 0);

    // precompute the bin center positions for x, y, T
    m_x_array = boost::shared_array<float>(new float[m_nbins_x]);
    for (unsigned int i = 0; i < m_nbins_x; i++)
        {
        float x = -m_max_x + float(i) * m_dx;
        m_x_array[i] = x;
        }

    // precompute the bin center positions for y
    // what should this calc be?
    m_y_array = boost::shared_array<float>(new float[m_nbins_y]);
    for (unsigned int i = 0; i < m_nbins_y; i++)
        {
        float y = -m_max_y + float(i) * m_dy;
        m_y_array[i] = y;
        }

    m_T_array = boost::shared_array<float>(new float[m_nbins_T]);
    for (unsigned int i = 0; i < m_nbins_T; i++)
        {
        float T = -m_max_T + float(i) * m_dT;
        m_T_array[i] = T;
        }

    // precompute the bin center positions for x
    // what should this calc be?

    if (useCells())
        {
        m_lc = new locality::LinkCell(box, sqrtf(max_x*max_x + max_y*max_y));
        }
    }

PMFTXYT2D::~PMFTXYT2D()
    {
    if(useCells())
    delete m_lc;
    }

class ComputePMFTXYT2DWithoutCellList
    {
    private:
        atomic<unsigned int> *m_pcf_array;
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        unsigned int m_nbins_T;
        const trajectory::Box m_box;
        const float m_max_x;
        const float m_max_y;
        const float m_max_T;
        const float m_dx;
        const float m_dy;
        const float m_dT;
        const float3 *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        const float3 *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFTXYT2DWithoutCellList(atomic<unsigned int> *pcf_array,
                                   unsigned int nbins_x,
                                   unsigned int nbins_y,
                                   unsigned int nbins_T,
                                   const trajectory::Box &box,
                                   const float max_x,
                                   const float max_y,
                                   const float max_T,
                                   const float dx,
                                   const float dy,
                                   const float dT,
                                   const float3 *ref_points,
                                   float *ref_orientations,
                                   unsigned int Nref,
                                   const float3 *points,
                                   float *orientations,
                                   unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_nbins_T(nbins_T), m_box(box),
              m_max_x(max_x), m_max_y(max_y), m_max_T(max_T), m_dx(dx), m_dy(dy), m_dT(dT), m_ref_points(ref_points), m_ref_orientations(orientations),
              m_Nref(Nref), m_points(points), m_orientations(orientations), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            float dx_inv = 1.0f / m_dx;
            float maxxsq = m_max_x * m_max_x;
            float dy_inv = 1.0f / m_dy;
            float maxysq = m_max_y * m_max_y;
            float dT_inv = 1.0f / m_dT;

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                float3 ref = m_ref_points[i];
                for (unsigned int j = 0; j < m_Np; j++)
                    {
                    if (i == j)
                        {
                        continue;
                        }
                    float3 point = m_points[j];
                    // compute r between the two particles
                    float dx = float(point.x - ref.x);
                    float dy = float(point.y - ref.y);

                    float3 delta = m_box.wrap(make_float3(dx, dy, (float)0));

                    float xsq = delta.x*delta.x;
                    float ysq = delta.y*delta.y;
                    if ((xsq < maxxsq) && (ysq < maxysq))
                        {
                        // rotate interparticle vector
                        vec2<Scalar> myVec(delta.x, delta.y);
                        rotmat2<Scalar> myMat(-m_ref_orientations[i]);
                        vec2<Scalar> rotVec = myMat * myVec;
                        float x = rotVec.x;
                        float y = rotVec.y;
                        float T = m_orientations[j] - m_ref_orientations[i];
                        // float x = delta.x;
                        // float y = delta.y;

                        // bin that point
                        float binx = (m_nbins_x / 2) + (x * dx_inv);
                        float biny = (m_nbins_y / 2) + (y * dy_inv);
                        float binT = (m_nbins_T / 2) + (T * dT_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                        unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                        unsigned int ibinT = _mm_cvtt_ss2si(_mm_load_ss(&binT));
                        #else
                        unsigned int ibinx = (unsigned int)(binx);
                        unsigned int ibiny = (unsigned int)(biny);
                        unsigned int ibinT = (unsigned int)(binT);
                        #endif

                        if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y) && (ibinT < m_nbins_T))
                            {
                            // m_pcf_array[ibiny*m_nbins_x*m_nbins_T + ibinx*m_nbins_T + ibinT]++;
                            m_pcf_array[ibinT*m_nbins_y*m_nbins_x + ibiny*m_nbins_x + ibinx]++;
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

class ComputePMFTXYT2DWithCellList
    {
    private:
        atomic<unsigned int> *m_pcf_array;
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        unsigned int m_nbins_T;
        const trajectory::Box m_box;
        const float m_max_x;
        const float m_max_y;
        const float m_max_T;
        const float m_dx;
        const float m_dy;
        const float m_dT;
        const locality::LinkCell *m_lc;
        float3 *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        float3 *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFTXYT2DWithCellList(atomic<unsigned int> *pcf_array,
                                unsigned int nbins_x,
                                unsigned int nbins_y,
                                unsigned int nbins_T,
                                const trajectory::Box &box,
                                const float max_x,
                                const float max_y,
                                const float max_T,
                                const float dx,
                                const float dy,
                                const float dT,
                                const locality::LinkCell *lc,
                                float3 *ref_points,
                                float *ref_orientations,
                                unsigned int Nref,
                                float3 *points,
                                float *orientations,
                                unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_nbins_T(nbins_T), m_box(box),
              m_max_x(max_x), m_max_y(max_y), m_max_T(max_T), m_dx(dx), m_dy(dy), m_dT(dT), m_lc(lc),
              m_ref_points(ref_points), m_ref_orientations(orientations), m_Nref(Nref), m_points(points), m_orientations(orientations), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_points);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            float dx_inv = 1.0f / m_dx;
            float maxxsq = m_max_x * m_max_x;
            float dy_inv = 1.0f / m_dy;
            float maxysq = m_max_y * m_max_y;
            float dT_inv = 1.0f / m_dT;

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                // get the cell the point is in
                float3 ref = m_ref_points[i];
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
                        // compute r between the two particles
                        // will skip same particle
                        if (i == j)
                            {
                            continue;
                            }
                        float3 point = m_points[j];
                        float dx = float(point.x - ref.x);
                        float dy = float(point.y - ref.y);
                        float3 delta = m_box.wrap(make_float3(dx, dy, (float)0));

                        float xsq = delta.x*delta.x;
                        float ysq = delta.y*delta.y;
                        if ((xsq < maxxsq) && (ysq < maxysq))
                            {
                            // rotate interparticle vector
                            vec2<Scalar> myVec(delta.x, delta.y);
                            rotmat2<Scalar> myMat(-m_ref_orientations[i]);
                            vec2<Scalar> rotVec = myMat * myVec;
                            float x = rotVec.x;
                            float y = rotVec.y;
                            float T = m_orientations[j] - m_ref_orientations[i];
                            // float x = delta.x;
                            // float y = delta.y;

                            // bin that point
                            float binx = (m_nbins_x / 2) + (x * dx_inv);
                            float biny = (m_nbins_y / 2) + (y * dy_inv);
                            float binT = (m_nbins_T / 2) + (T * dT_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                            unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                            unsigned int ibinT = _mm_cvtt_ss2si(_mm_load_ss(&binT));
                            #else
                            unsigned int ibinx = (unsigned int)(binx);
                            unsigned int ibiny = (unsigned int)(biny);
                            unsigned int ibinT = (unsigned int)(binT);
                            #endif

                            if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y) && (ibinT < m_nbins_T))
                                {
                                // m_pcf_array[ibiny*m_nbins_x*m_nbins_T + ibinx*m_nbins_T + ibinT]++;
                                m_pcf_array[ibinT*m_nbins_y*m_nbins_x + ibiny*m_nbins_x + ibinx]++;
                                }
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

bool PMFTXYT2D::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());

    if (!m_box.is2D())
        l_min = fmin(l_min, m_box.getLz());

    float rmax = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y);

    if (rmax < l_min/3.0f)
        return true;

    return false;
    }

void PMFTXYT2D::compute(unsigned int *pcf_array,
                        float3 *ref_points,
                        float *ref_orientations,
                        unsigned int Nref,
                        float3 *points,
                        float *orientations,
                        unsigned int Np)
    {
    if (useCells())
        {
        m_lc->computeCellList(points, Np);
        parallel_for(blocked_range<size_t>(0,Nref), ComputePMFTXYT2DWithCellList((atomic<unsigned int>*)pcf_array,
                                                                            m_nbins_x,
                                                                            m_nbins_y,
                                                                            m_nbins_T,
                                                                            m_box,
                                                                            m_max_x,
                                                                            m_max_y,
                                                                            m_max_T,
                                                                            m_dx,
                                                                            m_dy,
                                                                            m_dT,
                                                                            m_lc,
                                                                            ref_points,
                                                                            ref_orientations,
                                                                            Nref,
                                                                            points,
                                                                            orientations,
                                                                            Np));
        }
    else
        {
        printf("not using cells\n");
        fflush(stdout);
        parallel_for(blocked_range<size_t>(0,Nref), ComputePMFTXYT2DWithoutCellList((atomic<unsigned int>*)pcf_array,
                                                                               m_nbins_x,
                                                                               m_nbins_y,
                                                                               m_nbins_T,
                                                                               m_box,
                                                                               m_max_x,
                                                                               m_max_y,
                                                                               m_max_T,
                                                                               m_dx,
                                                                               m_dy,
                                                                               m_dT,
                                                                               ref_points,
                                                                               ref_orientations,
                                                                               Nref,
                                                                               points,
                                                                               orientations,
                                                                               Np));
        }
    }

void PMFTXYT2D::computePy(boost::python::numeric::array pcf_array,
                          boost::python::numeric::array ref_points,
                          boost::python::numeric::array ref_orientations,
                          boost::python::numeric::array points,
                          boost::python::numeric::array orientations)
    {
    // validate input type and rank
    num_util::check_type(pcf_array, PyArray_INT);
    num_util::check_rank(pcf_array, 3);
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(ref_orientations, PyArray_FLOAT);
    num_util::check_rank(ref_orientations, 1);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(orientations, PyArray_FLOAT);
    num_util::check_rank(orientations, 1);

    // validate array dims
    num_util::check_dim(pcf_array, 0, m_nbins_T);
    num_util::check_dim(pcf_array, 1, m_nbins_y);
    num_util::check_dim(pcf_array, 2, m_nbins_x);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];

    // check the size of angles to be correct
    num_util::check_dim(ref_orientations, 0, Nref);
    // num_util::check_dim(ref_orientations, 0, 1);
    num_util::check_dim(orientations, 0, Np);

    // get the raw data pointers and compute the cell list
    unsigned int* pcf_array_raw = (unsigned int*) num_util::data(pcf_array);
    float3* ref_points_raw = (float3*) num_util::data(ref_points);
    float* ref_orientations_raw = (float*) num_util::data(ref_orientations);
    float3* points_raw = (float3*) num_util::data(points);
    float* orientations_raw = (float*) num_util::data(orientations);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(pcf_array_raw,
                ref_points_raw,
                ref_orientations_raw,
                Nref,
                points_raw,
                orientations_raw,
                Np);
        }
    }

void export_PMFTXYT2D()
    {
    class_<PMFTXYT2D>("PMFTXYT2D", init<trajectory::Box&, float, float, float, float, float, float>())
        .def("getBox", &PMFTXYT2D::getBox, return_internal_reference<>())
        .def("compute", &PMFTXYT2D::computePy)
        .def("getX", &PMFTXYT2D::getXPy)
        .def("getY", &PMFTXYT2D::getYPy)
        .def("getT", &PMFTXYT2D::getTPy)
        ;
    }

}; }; // end namespace freud::pmft
