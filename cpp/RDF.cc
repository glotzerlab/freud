#include <stdexcept>

#include "RDF.h"

using namespace std;
using namespace boost::python;

RDF::RDF(const Box& box, float rmax, float dr)
    : m_box(box), m_rmax(rmax), m_dr(dr), m_lc(box, rmax)
    {
    if (dr < 0.0f)
        throw invalid_argument("dr must be positive");
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (dr > rmax)
        throw invalid_argument("rmax must be greater than dr");
    if (!(box.getLx() >= 3.0 * m_rmax && box.getLy() >= 3.0 * m_rmax && box.getLz() >= 3.0 * m_rmax))
        throw invalid_argument("RDF currently does not support computations where rmax > 1/3 any box dimension");
    
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
        m_vol_array[i] = 4.0f / 3.0f * M_PI * (r*r*r - prevr*prevr*prevr);
        }
    }

void RDF::compute(float *x_ref,
                  float *y_ref,
                  float *z_ref,
                  unsigned int Nref,
                  float *x,
                  float *y,
                  float *z,
                  unsigned int Np)
    {
    assert(x_ref);
    assert(y_ref);
    assert(z_ref);
    assert(Nref > 0);
    assert(x);
    assert(y);
    assert(z);
    assert(Np > 0);
    
    // bin the x,y,z particles
    m_lc.computeCellList(x, y, z, Np);
    
    // zero the bin counts for totalling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    
    // for each reference point
    for (unsigned int i = 0; i < Nref; i++)
        {
        // get the cell the point is in
        unsigned int ref_cell = m_lc.getCell(x_ref[i], y_ref[i], z_ref[i]);
        
        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];
            
            // iterate over the particles in that cell
            LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                {
                // compute r between the two particles
                float dx = float(x_ref[i] - x[j]);
                float dy = float(y_ref[i] - y[j]);
                float dz = float(z_ref[i] - z[j]);
                m_box.wrap(dx, dy, dz);
                
                float rsq = dx*dx + dy*dy + dz*dz;
                float r = sqrtf(rsq);
                
                // bin that r
                unsigned int bin = (unsigned int)(floorf(r / m_dr));
                if (bin < m_nbins)
                    {
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

void RDF::computePy(boost::python::numeric::array x_ref,
                    boost::python::numeric::array y_ref,
                    boost::python::numeric::array z_ref,
                    boost::python::numeric::array x,
                    boost::python::numeric::array y,
                    boost::python::numeric::array z)
    {
    // validate input type and rank
    num_util::check_type(x, PyArray_FLOAT);
    num_util::check_rank(x, 1);
    num_util::check_type(y, PyArray_FLOAT);
    num_util::check_rank(y, 1);
    num_util::check_type(z, PyArray_FLOAT);
    num_util::check_rank(z, 1);
    num_util::check_type(x_ref, PyArray_FLOAT);
    num_util::check_rank(x_ref, 1);
    num_util::check_type(y_ref, PyArray_FLOAT);
    num_util::check_rank(y_ref, 1);
    num_util::check_type(z_ref, PyArray_FLOAT);
    num_util::check_rank(z_ref, 1);
    
    // validate all inputs are the same size
    unsigned int Np = num_util::size(x);
    num_util::check_size(y, Np);
    num_util::check_size(z, Np);
    unsigned int Nref = num_util::size(x_ref);
    num_util::check_size(y_ref, Nref);
    num_util::check_size(z_ref, Nref);
    
    // get the raw data pointers and compute the cell list
    float* x_raw = (float*) num_util::data(x);
    float* y_raw = (float*) num_util::data(y);
    float* z_raw = (float*) num_util::data(z);
    float* x_ref_raw = (float*) num_util::data(x_ref);
    float* y_ref_raw = (float*) num_util::data(y_ref);
    float* z_ref_raw = (float*) num_util::data(z_ref);

    compute(x_ref_raw, y_ref_raw, z_ref_raw, Nref, x_raw, y_raw, z_raw, Np);
    }

void export_RDF()
    {
    class_<RDF>("RDF", init<Box&, float, float>())
        .def("getBox", &RDF::getBox, return_internal_reference<>())
        .def("compute", &RDF::computePy)
        .def("getRDF", &RDF::getRDFPy)
        .def("getR", &RDF::getRPy)
        ;
    }