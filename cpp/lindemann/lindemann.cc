#include "lindemann.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost::python;

/*! \file lind.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace lindemann {

Lind::Lind(const trajectory::Box& box, float rmax, float dr)
    : m_box(box), m_rmax(rmax), m_dr(dr)
    {
    if (dr < 0.0f)
        throw invalid_argument("dr must be positive");
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (dr > rmax)
        throw invalid_argument("rmax must be greater than dr");
    if (rmax > box.getLx()/2 || rmax > box.getLy()/2)
        throw invalid_argument("rmax must be smaller than half the smallest box size");
    if (rmax > box.getLz()/2 && !box.is2D())
        throw invalid_argument("rmax must be smaller than half the smallest box size");

    // m_nbins = int(floorf(m_rmax / m_dr));
    // assert(m_nbins > 0);
    // m_rdf_array = boost::shared_array<float>(new float[m_nbins]);
    // memset((void*)m_rdf_array.get(), 0, sizeof(float)*m_nbins);
    // m_bin_counts = boost::shared_array<unsigned int>(new unsigned int[m_nbins]);
    // memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    // m_N_r_array = boost::shared_array<float>(new float[m_nbins]);
    // memset((void*)m_N_r_array.get(), 0, sizeof(unsigned int)*m_nbins);

    // precompute the bin center positions
    // m_r_array = boost::shared_array<float>(new float[m_nbins]);
    // for (unsigned int i = 0; i < m_nbins; i++)
    //     {
    //     float r = float(i) * m_dr;
    //     float nextr = float(i+1) * m_dr;
    //     m_r_array[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
    //     }

    // precompute cell volumes
    // m_vol_array = boost::shared_array<float>(new float[m_nbins]);
    // for (unsigned int i = 0; i < m_nbins; i++)
    //     {
    //     float r = float(i) * m_dr;
    //     float nextr = float(i+1) * m_dr;
    //     if (m_box.is2D())
    //         m_vol_array[i] = M_PI * (nextr*nextr - r*r);
    //     else
    //         m_vol_array[i] = 4.0f / 3.0f * M_PI * (nextr*nextr*nextr - r*r*r);
    //     }

    if (useCells())
        {
        m_lc = new locality::LinkCell(box, rmax);
        }
    }

Lind::~Lind()
    {
    if(useCells())
    delete m_lc;
    }

bool Lind::useCells()
    {
    return false;
    }

void Lind::compute(const float3 *points,
                 unsigned int Np,
                 unsigned int Nf)
    {
    computeWithoutCellList(points, Np, Nf);
    }

void Lind::computeWithoutCellList(const float3 *points,
                 unsigned int Np,
                 unsigned int Nf)
    {
    // zero the bin counts for totaling
    // memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;

    // for each reference point
    float lindex;
    for (unsigned int i = 0; i < Np; i++)
        {
        lindex = 0;
        for (unsigned int j = 0; j < Np; j++)
            {
            // avoid calling on same particle
            if (i == j)
                {
                continue;
                }
            float r_ij = 0;
            float rsq_ij = 0;
            for (unsigned int k = 0; k < Nf; k++)
                {
                // compute r between the two particles
                float dx = float(points[k * Np + i].x - points[k * Np + j].x);
                float dy = float(points[k * Np + i].y - points[k * Np + j].y);
                float dz = float(points[k * Np + i].z - points[k * Np + j].z);

                float3 delta = m_box.wrap(make_float3(dx, dy, dz));

                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                float r = sqrtf(rsq);
                r_ij += r;
                rsq_ij += rsq;
                } // done looping over frames
            float avg_r_ij = r_ij / ((float) Nf);
            float avg_rsq_ij = rsq_ij / ((float) Nf);
            float tmp_lindex = (sqrtf(abs(avg_rsq_ij - (avg_r_ij * avg_r_ij))) / avg_r_ij);
            // if (tmp_lindex != tmp_lindex)
            //     {
            //     printf("fucking nan detected\n");
            //     printf("r_ij = %f\n", r_ij);
            //     printf("rsq_ij = %f\n", rsq_ij);
            //     printf("avg_r_ij = %f\n", avg_r_ij);
            //     printf("avg_rsq_ij = %f\n", avg_rsq_ij);
            //     printf("inner sqrtf = %f\n", (avg_rsq_ij - (avg_r_ij * avg_r_ij)));
            //     printf("inner sqrtf = %f\n", tmp_lindex);
            //     }
            lindex += tmp_lindex;
            }
        lindex = (1.0 / (((float) Np) - 1.0)) * lindex;
        m_lindex += lindex;
        } // done looping over reference points
    // calc Lindexmann Index
    m_lindex /= Np;
    }

void Lind::computePy(boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 3);

    // validate that the 3rd dimension is only 3
    num_util::check_dim(points, 2, 3);
    // Nf = number of frames
    // Np = number of particles
    unsigned int Nf = num_util::shape(points)[0];
    unsigned int Np = num_util::shape(points)[1];

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np, Nf);
        }
    }

void export_lindemann()
    {
    class_<Lind>("Lind", init<trajectory::Box&, float, float>())
        .def("getBox", &Lind::getBox, return_internal_reference<>())
        .def("compute", &Lind::computePy)
        .def("getLindex", &Lind::getLindexPy)
        ;
    }

}; }; // end namespace freud::lindemann
