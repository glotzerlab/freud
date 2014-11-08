#include "PMFXY2D.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

#define swap freud_swap
#include "VectorMath.h"
#undef swap

using namespace std;
using namespace boost::python;

using namespace tbb;

/*! \internal
    \file PMFXY2D.cc
    \brief Routines for computing 2D anisotropic potential of mean force
*/

namespace freud { namespace pmft {

PMFXY2D::PMFXY2D(float max_x, float max_y, float dx, float dy)
    : m_box(trajectory::Box()), m_max_x(max_x), m_max_y(max_y), m_dx(dx), m_dy(dy)
    {
    if (dx < 0.0f)
        throw invalid_argument("dx must be positive");
    if (dy < 0.0f)
        throw invalid_argument("dy must be positive");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    if (dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (dy > max_y)
        throw invalid_argument("max_y must be greater than dy");

    m_nbins_x = int(2 * floorf(m_max_x / m_dx));
    assert(m_nbins_x > 0);
    m_nbins_y = int(2 * floorf(m_max_y / m_dy));
    assert(m_nbins_y > 0);

    // precompute the bin center positions for x
    m_x_array = boost::shared_array<float>(new float[m_nbins_x]);
    for (unsigned int i = 0; i < m_nbins_x; i++)
        {
        float x = float(i) * m_dx;
        float nextx = float(i+1) * m_dx;
        m_x_array[i] = -m_max_x + ((x + nextx) / 2.0);
        }

    // precompute the bin center positions for y
    m_y_array = boost::shared_array<float>(new float[m_nbins_y]);
    for (unsigned int i = 0; i < m_nbins_y; i++)
        {
        float y = float(i) * m_dy;
        float nexty = float(i+1) * m_dy;
        m_y_array[i] = -m_max_y + ((y + nexty) / 2.0);
        }

    // create and populate the pcf_array
    m_pcf_array = boost::shared_array<unsigned int>(new unsigned int[m_nbins_x * m_nbins_y]);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);

    m_lc = new locality::LinkCell();
    }

PMFXY2D::~PMFXY2D()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_pcf_array.begin(); i != m_local_pcf_array.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_lc;
    }

void PMFXY2D::updateBox(trajectory::Box& box)
    {
    // check to make sure the provided box is valid
    if (m_max_x > box.getLx()/2 || m_max_y > box.getLy()/2)
        throw invalid_argument("rmax must be smaller than half the smallest box size");
    if (!box.is2D())
        throw invalid_argument("box must be 2D");
    // see if it is different than the current box
    if (m_box != box)
        {
        m_box = box;
        float max_val = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y);
        m_lc->updateBox(m_box, max_val);
        }
    }

class CombinePCFXY2D
    {
    private:
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        unsigned int *m_pcf_array;
        tbb::enumerable_thread_specific<unsigned int *>& m_local_pcf_array;
    public:
        CombinePCFXY2D(unsigned int nbins_x,
                       unsigned int nbins_y,
                       unsigned int *pcf_array,
                       tbb::enumerable_thread_specific<unsigned int *>& local_pcf_array)
            : m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_pcf_array(pcf_array), m_local_pcf_array(local_pcf_array)
        {
        }
        void operator()( const blocked_range<size_t> &myBin ) const
            {
            Index2D b_i = Index2D(m_nbins_x, m_nbins_y);
            for (size_t i = myBin.begin(); i != myBin.end(); i++)
                {
                for (size_t j = 0; j < m_nbins_y; j++)
                    {
                    for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_pcf_array.begin();
                         local_bins != m_local_pcf_array.end(); ++local_bins)
                        {
                        m_pcf_array[b_i((int)i, (int)j)] += (*local_bins)[b_i((int)i, (int)j)];
                        }
                    }
                }
            }
    };

//! \internal
/*! \brief Helper class to compute PMF in parallel with the cell list
*/

class ComputePMFXY2D
    {
    private:
        tbb::enumerable_thread_specific<unsigned int *>& m_pcf_array;
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        const trajectory::Box m_box;
        const float m_max_x;
        const float m_max_y;
        const float m_dx;
        const float m_dy;
        const locality::LinkCell *m_lc;
        vec3<float> *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        vec3<float> *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFXY2D(tbb::enumerable_thread_specific<unsigned int *>& pcf_array,
                       unsigned int nbins_x,
                       unsigned int nbins_y,
                       const trajectory::Box &box,
                       const float max_x,
                       const float max_y,
                       const float dx,
                       const float dy,
                       const locality::LinkCell *lc,
                       vec3<float> *ref_points,
                       float *ref_orientations,
                       unsigned int Nref,
                       vec3<float> *points,
                       float *orientations,
                       unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_box(box),
              m_max_x(max_x), m_max_y(max_y), m_dx(dx), m_dy(dy), m_lc(lc), m_ref_points(ref_points),
              m_ref_orientations(ref_orientations), m_Nref(Nref), m_points(points), m_orientations(orientations),
              m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_points);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            // precalc some values for faster computation within the loop
            float dx_inv = 1.0f / m_dx;
            float maxxsq = m_max_x * m_max_x;
            float dy_inv = 1.0f / m_dy;
            float maxysq = m_max_y * m_max_y;

            Index2D b_i = Index2D(m_nbins_x, m_nbins_y);

            bool exists;
            m_pcf_array.local(exists);
            if (! exists)
                {
                m_pcf_array.local() = new unsigned int [m_nbins_x*m_nbins_y];
                memset((void*)m_pcf_array.local(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);
                }

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                vec3<float> ref = m_ref_points[i];
                // get the cell the point is in
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

                        // check that the particle is not checking itself
                        // 1e-6 is an arbitrary value that could be set differently if needed
                        if (rsq < 1e-6)
                            {
                            continue;
                            }

                        // rotate interparticle vector
                        vec2<float> myVec(delta.x, delta.y);
                        rotmat2<float> myMat = rotmat2<float>::fromAngle(-m_ref_orientations[i]);
                        vec2<float> rotVec = myMat * myVec;
                        float x = rotVec.x + m_max_x;
                        float y = rotVec.y + m_max_y;

                        // find the bin to increment
                        float binx = floorf(x * dx_inv);
                        float biny = floorf(y * dy_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                        unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                        #else
                        unsigned int ibinx = (unsigned int)(binx);
                        unsigned int ibiny = (unsigned int)(biny);
                        #endif

                        // increment the bin
                        if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y))
                            {
                            ++m_pcf_array.local()[b_i(ibinx, ibiny)];
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
*/

void PMFXY2D::resetPCF()
    {
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);
    }

//! \internal
/*! \brief Helper functionto direct the calculation to the correct helper class
*/

void PMFXY2D::compute(vec3<float> *ref_points,
                      float *ref_orientations,
                      unsigned int Nref,
                      vec3<float> *points,
                      float *orientations,
                      unsigned int Np)
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_pcf_array.begin(); i != m_local_pcf_array.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);
        }
    m_lc->computeCellList(points, Np);
    parallel_for(blocked_range<size_t>(0,Nref),
                 ComputePMFXY2D(m_local_pcf_array,
                                m_nbins_x,
                                m_nbins_y,
                                m_box,
                                m_max_x,
                                m_max_y,
                                m_dx,
                                m_dy,
                                m_lc,
                                ref_points,
                                ref_orientations,
                                Nref,
                                points,
                                orientations,
                                Np));
    parallel_for(blocked_range<size_t>(0,m_nbins_x),
                 CombinePCFXY2D(m_nbins_x,
                                m_nbins_y,
                                m_pcf_array.get(),
                                m_local_pcf_array));
    }

//! \internal
/*! \brief Exposed function to python to calculate the PMF
*/

void PMFXY2D::computePy(trajectory::Box& box,
                        boost::python::numeric::array ref_points,
                        boost::python::numeric::array ref_orientations,
                        boost::python::numeric::array points,
                        boost::python::numeric::array orientations)
    {
    // validate input type and rank
    updateBox(box);
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(ref_orientations, PyArray_FLOAT);
    num_util::check_rank(ref_orientations, 1);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(orientations, PyArray_FLOAT);
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
        compute(ref_points_raw,
                ref_orientations_raw,
                Nref,
                points_raw,
                orientations_raw,
                Np);
        }
    }

void export_PMFXY2D()
    {
    class_<PMFXY2D>("PMFXY2D", init<float, float, float, float>())
        .def("getBox", &PMFXY2D::getBox, return_internal_reference<>())
        .def("compute", &PMFXY2D::computePy)
        .def("getPCF", &PMFXY2D::getPCFPy)
        .def("getX", &PMFXY2D::getXPy)
        .def("getY", &PMFXY2D::getYPy)
        .def("resetPCF", &PMFXY2D::resetPCFPy)
        ;
    }

}; }; // end namespace freud::pmft
