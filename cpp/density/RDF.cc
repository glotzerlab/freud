#include "RDF.h"

#include <stdexcept>
#include <emmintrin.h>

using namespace std;
using namespace boost::python;

namespace freud { namespace density {

RDF::RDF(const trajectory::Box& box, float rmax, float dr)
    : m_box(box), m_rmax(rmax), m_dr(dr), m_lc(box, rmax)
    {
    if (dr < 0.0f)
        throw invalid_argument("dr must be positive");
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (dr > rmax)
        throw invalid_argument("rmax must be greater than dr");
    
    m_nbins = int(ceilf(m_rmax / m_dr));
    assert(m_nbins > 0);
    m_rdf_array = boost::shared_array<float>(new float[m_nbins]);
    memset((void*)m_rdf_array.get(), 0, sizeof(float)*m_nbins);
    m_bin_counts = boost::shared_array<unsigned int>(new unsigned int[m_nbins]);
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    
    // precompute the bin start positions
    m_r_array = boost::shared_array<float>(new float[m_nbins]);
    for (unsigned int i = 0; i < m_nbins; i++)
        m_r_array[i] = float(i) * m_dr;
    
    // precompute cell volumes
    m_vol_array = boost::shared_array<float>(new float[m_nbins]);
    m_vol_array[0] = 0.0f;
    for (unsigned int i = 1; i < m_nbins; i++)
        {
        float r = float(i) * m_dr;
        float prevr = float(i-1) * m_dr;
        if (m_box.is2D())
            m_vol_array[i] = M_PI * (r*r - prevr*prevr);
        else
            m_vol_array[i] = 4.0f / 3.0f * M_PI * (r*r*r - prevr*prevr*prevr);
        }
    }

void RDF::compute(const float3 *ref_points,
                  unsigned int Nref,
                  const float3 *points,
                  unsigned int Np)
    {
    assert(ref_points);
    assert(points);
    assert(Nref > 0);
    assert(Np > 0);
    
    // bin the x,y,z particles
    m_lc.computeCellList(points, Np);
    
    // zero the bin counts for totalling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;
    
    // for each reference point
    for (unsigned int i = 0; i < Nref; i++)
        {
        // get the cell the point is in
        float3 ref = ref_points[i];
        unsigned int ref_cell = m_lc.getCell(ref);
        
        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];
            
            // iterate over the particles in that cell
            locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                {
                // compute r between the two particles
                float dx = float(ref.x - points[j].x);
                float dy = float(ref.y - points[j].y);
                float dz = float(ref.z - points[j].z);
                float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                
                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                if (rsq < rmaxsq)
                    {
                    float r = sqrtf(rsq);
                
                    // bin that r
                    float binr = r * dr_inv;
                    // fast float to int conversion with truncation
                    unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                    m_bin_counts[bin]++;
                    }
                }
            }
        }
    
    // done looping over reference points
    // now compute the rdf
    float ndens = float(Np) / m_box.getVolume();
    m_rdf_array[0] = 0;
    for (unsigned int bin = 1; bin < m_nbins; bin++)
        {
        float avg_counts = m_bin_counts[bin] / float(Nref);
        m_rdf_array[bin] = avg_counts / m_vol_array[bin] / ndens;
        }
    }

void RDF::computePy(boost::python::numeric::array ref_points,
                    boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];
    
    // get the raw data pointers and compute the cell list
    float3* ref_points_raw = (float3*) num_util::data(ref_points);
    float3* points_raw = (float3*) num_util::data(points);

    compute(ref_points_raw, Nref, points_raw, Np);
    }

void export_RDF()
    {
    class_<RDF>("RDF", init<trajectory::Box&, float, float>())
        .def("getBox", &RDF::getBox, return_internal_reference<>())
        .def("compute", &RDF::computePy)
        .def("getRDF", &RDF::getRDFPy)
        .def("getR", &RDF::getRPy)
        ;
    }

}; }; // end namespace freud::density