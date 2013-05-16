#include "complement.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost::python;

namespace freud { namespace complement {

complement::complement(const trajectory::Box& box, float rmax, float dr)
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

    m_nbins = int(floorf(m_rmax / m_dr));
    assert(m_nbins > 0);
    m_rdf_array = boost::shared_array<float>(new float[m_nbins]);
    memset((void*)m_rdf_array.get(), 0, sizeof(float)*m_nbins);
    m_bin_counts = boost::shared_array<unsigned int>(new unsigned int[m_nbins]);
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    m_N_r_array = boost::shared_array<float>(new float[m_nbins]);
    memset((void*)m_N_r_array.get(), 0, sizeof(unsigned int)*m_nbins);
    
    // precompute the bin center positions
    m_r_array = boost::shared_array<float>(new float[m_nbins]);
    for (unsigned int i = 0; i < m_nbins; i++)
    {
    float r = float(i) * m_dr;
    float nextr = float(i+1) * m_dr;
    m_r_array[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
    }
    
    // precompute cell volumes
    m_vol_array = boost::shared_array<float>(new float[m_nbins]);
    for (unsigned int i = 0; i < m_nbins; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        if (m_box.is2D())
            m_vol_array[i] = M_PI * (nextr*nextr - r*r);
        else
            m_vol_array[i] = 4.0f / 3.0f * M_PI * (nextr*nextr*nextr - r*r*r);
        }

    if (useCells())
    {
    m_lc = new locality::LinkCell(box, rmax);
    }
    }

complement::~complement()
    {
    if(useCells())
    delete m_lc;
    }

bool complement::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());
    if (m_box.is2D())
    l_min = fmin(l_min, m_box.getLy());
    if (m_rmax < l_min/3)
    return true;
    return false;
    }

bool complement::sameSide(float3 A, float3 B, float3 r, float3 p)
    {
    float3 BA;
    float3 rA;
    float3 pA;
    
    BA.x = B.x - A.x;
    BA.y = B.y - A.y;
    BA.z = B.z - A.z;
    
    rA.x = r.x - A.x;
    rA.y = r.y - A.y;
    rA.z = r.z - A.z;
    
    pA.x = p.x - A.x;
    pA.y = p.y - A.y;
    pA.z = p.z - A.z;
    
    float3 ref = cross(BA, rA);
    float3 test = cross(BA, pA);
    if (dot(ref, test) >= 0)
        {
        return true;
        }
    else
        {
        return false;
        }
    }

bool complement::isInside(float2 t[], float2 p)
    {
    float3 A;
    float3 B;
    float3 C;
    float3 P;
    
    A.x = t[0].x;
    A.y = t[0].y;
    A.z = 0;
    
    B.x = t[1].x;
    B.y = t[1].y;
    B.z = 0;
    
    C.x = t[2].x;
    C.y = t[2].y;
    C.z = 0;
    
    P.x = p.x;
    P.y = p.y;
    P.z = 0;
    
    bool BC = sameSide(B, C, A, P);
    bool AC = sameSide(A, C, B, P);
    bool AB = sameSide(A, B, C, P);
    
    if (AB && BC && AC)
        {
        return true;
        }
    else
        {
        return false;
        }
    
    }

float3 complement::cross(float3 v1, float3 v2)
    {
    float3 v;
    v.x = (v1.y * v2.z) - (v2.y * v1.z);
    v.y = (v2.x * v1.z) - (v1.x * v2.z);
    v.z = (v1.x * v2.y) - (v2.x * v1.y);
    return v;
    }
    
float complement::dot(float3 v1, float3 v2)
    {
    float v;
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    }

float2 complement::mat_rotate(float2 point, float angle)
    {
    float2 rot;
    float mysin = sinf(angle);
    float mycos = cosf(angle);
    rot.x = mycos * point.x + -mysin * point.y;
    rot.y = mysin * point.x + mycos * point.y;
    return rot;
    }
    
float2 complement::into_local(float3 ref_point,
                            float3 point,
                            float2 vert,
                            float ref_angle,
                            float angle)
    {
    float2 local;
    // I think this may be backward
    // Currently will only do 2 dimensions
    local = mat_rotate(mat_rotate(vert, -ref_angle), angle);
    local.x = local.x + (ref_point.x - point.x);
    local.y = local.y + (ref_point.y - point.y);
    return local;
    }

float complement::cavity_depth(float2 t[])
    {
    float m = (t[2].y - t[0].y)/(t[2].x - t[0].x);
    float m_inv = (t[2].x - t[0].x)/(t[2].y - t[0].y);
    float2 intersect;
    intersect.x = (t[1].y - t[2].y + (t[1].x * m_inv) + (t[2].x * m)) * (1/(m + m_inv));
    intersect.y = m * (intersect.x - t[0].x) + t[0].y;
    float2 d_vec;
    d_vec.x = intersect.x - t[1].x;
    d_vec.y = intersect.y - t[1].y;
    float mag = intersect.x * intersect.x + intersect.y * intersect.y;
    return sqrt(mag);
    }

void complement::compute(const float3 *ref_points,
                  const float *ref_angles,
                  const float2 *ref_shape,
                  unsigned int *ref_verts,
                  unsigned int Nref,
                  unsigned int Nref_s,
                  unsigned int Nref_v,
                  const float3 *points,
                  const float *angles,
                  const float2 *shape,
                  unsigned int *verts,
                  unsigned int Np,
                  unsigned int Ns,
                  unsigned int Nv)
    {
    if (useCells())
    computeWithCellList(ref_points, ref_angles, ref_shape, ref_verts, Nref, Nref_s, Nref_v, points, angles, shape, verts, Np, Ns, Nv);
    else
    computeWithoutCellList(ref_points, ref_angles, ref_shape, ref_verts, Nref, Nref_s, Nref_v, points, angles, shape, verts, Np, Ns, Nv);
    }

void complement::computeWithoutCellList(const float3 *ref_points,
                  const float *ref_angles,
                  const float2 *ref_shape,
                  unsigned int *ref_verts,
                  unsigned int Nref,
                  unsigned int Nref_s,
                  unsigned int Nref_v,
                  const float3 *points,
                  const float *angles,
                  const float2 *shape,
                  unsigned int *verts,
                  unsigned int Np,
                  unsigned int Ns,
                  unsigned int Nv)
    {
    // zero the bin counts for totaling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;
    //unsigned int raw_cnt = 0;
    #pragma omp parallel
    {

    #pragma omp for schedule(guided)
    // for each reference point
    for (unsigned int i = 0; i < Nref; i++)
        {

        for (unsigned int j = 0; j < Np; j++)
            {
            // compute r between the two particles
            // New code here
            
            for (unsigned int k = 0; k < Nref_v; k++)
                {
                unsigned int tooth_index = ref_verts[k];
                float2 tooth = ref_shape[tooth_index];
                // I think this will test against all cavities, not just the one
                // If each were on k then it would be different
                for (unsigned int l = 0; l < Nv; l++)
                    {
                    unsigned int cavity_index = verts[l];
                    float2 cavity[3];
                    
                    cavity[0] = shape[cavity_index - 1];
                    cavity[1] = shape[cavity_index];
                    cavity[2] = shape[cavity_index + 1];
                    
                    float depth = cavity_depth(cavity);
                    
                    for (unsigned int m = 0; m < 3; m++)
                        {
                        cavity[m] = into_local(ref_points[i], points[j], cavity[m], ref_angles[i], angles[j]);
                        }
                    
                    bool test = isInside(cavity, tooth);
                    
                    if (test == true)
                        {
                        //raw_cnt++;
                        m_nmatch++;
                        // Calc the relative rdf
                        float dx = float(ref_points[i].x - points[j].x);
                        float dy = float(ref_points[i].y - points[j].y);
                        float dz = float(ref_points[i].z - points[j].z);
                        
                        float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                        
                        // Need to scale by cavity depth
                        
                        float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        if (rsq < rmaxsq)
                            {
                            float r = sqrtf(rsq);

                            // bin that r
                            float binr = r * dr_inv / depth;
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                                unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                            #else
                                unsigned int bin = (unsigned int)(binr);
                            #endif
                            #pragma omp atomic
                                m_bin_counts[bin]++;
                            }
                        }
                    }
                }
            }
        } // done looping over reference points
    } // End of parallel section

    // now compute the "rdf"
    float ndens = float(Np) / m_box.getVolume();
    m_rdf_array[0] = 0.0f;
    m_N_r_array[0] = 0.0f;
    m_N_r_array[1] = 0.0f;

    for (unsigned int bin = 1; bin < m_nbins; bin++)
        {
        float avg_counts = m_bin_counts[bin] / float(Nref);
        m_rdf_array[bin] = avg_counts / m_vol_array[bin] / ndens;

        if (bin+1 < m_nbins)
            m_N_r_array[bin+1] = m_N_r_array[bin] + avg_counts;
        }
    }

void complement::computeWithCellList(const float3 *ref_points,
                  const float *ref_angles,
                  const float2 *ref_shape,
                  unsigned int *ref_verts,
                  unsigned int Nref,
                  unsigned int Nref_s,
                  unsigned int Nref_v,
                  const float3 *points,
                  const float *angles,
                  const float2 *shape,
                  unsigned int *verts,
                  unsigned int Np,
                  unsigned int Ns,
                  unsigned int Nv)
    {
    assert(ref_points);
    assert(points);
    assert(Nref > 0);
    assert(Np > 0);
    
    // bin the x,y,z particles
    m_lc->computeCellList(points, Np);
    
    // zero the bin counts for totaling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;
    //unsigned int raw_cnt = 0;
    #pragma omp parallel
    {
    
    #pragma omp for schedule(guided)
    // for each reference point
    for (unsigned int i = 0; i < Nref; i++)
        {
        
        // get the cell the point is in
        float3 ref = ref_points[i];
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
                // I believe this is where I need to insert my code
                // compute r between the two particles
                for (unsigned int k = 0; k < Nref_v; k++)
                    {
                    unsigned int tooth_index = ref_verts[k];
                    float2 tooth = ref_shape[tooth_index];
                    
                    for (unsigned int l = 0; l < Nv; l++)
                        {
                        unsigned int cavity_index = verts[l];
                        float2 cavity [3];
                    
                        cavity[0] = shape[cavity_index - 1];
                        cavity[1] = shape[cavity_index];
                        cavity[2] = shape[cavity_index + 1];
                        
                        float depth = cavity_depth(cavity);
                        
                        for (unsigned int m = 0; m < 3; m++)
                            {
                            cavity[m] = into_local(ref_points[i], points[j], cavity[m], ref_angles[i], angles[j]);
                            }
                        
                        bool test = isInside(cavity, tooth);
                        
                        if (test == true)
                            {
                        
                            //raw_cnt++;
                            m_nmatch++;
                            // Calc the relative rdf
                            float dx = float(ref_points[i].x - points[j].x);
                            float dy = float(ref_points[i].y - points[j].y);
                            float dz = float(ref_points[i].z - points[j].z);
                        
                            float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                            float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        
                            if (rsq < rmaxsq)
                                {
                                float r = sqrtf(rsq);
                
                                // bin that r
                                float binr = r * dr_inv / depth;
                                // fast float to int conversion with truncation
                                #ifdef __SSE2__
                                    unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                                #else
                                    unsigned int bin = (unsigned int)(binr);
                                #endif
                                #pragma omp atomic
                                    m_bin_counts[bin]++;
                                }
                        
                            }
                        
                        }
                    
                    }
                }
            }
        } // done looping over reference points
    } // End of parallel section

    // now compute the rdf
    float ndens = float(Np) / m_box.getVolume();
    m_rdf_array[0] = 0.0f;
    m_N_r_array[0] = 0.0f;
    m_N_r_array[1] = 0.0f;

    for (unsigned int bin = 1; bin < m_nbins; bin++)
        {
        float avg_counts = m_bin_counts[bin] / float(Nref);
        m_rdf_array[bin] = avg_counts / m_vol_array[bin] / ndens;
        
        if (bin+1 < m_nbins)
            {
            m_N_r_array[bin+1] = m_N_r_array[bin] + avg_counts;
            }
        }
    }

void complement::computePy(boost::python::numeric::array ref_points,
                    boost::python::numeric::array ref_angles,
                    boost::python::numeric::array ref_shape,
                    boost::python::numeric::array ref_verts,
                    boost::python::numeric::array points,
                    boost::python::numeric::array angles,
                    boost::python::numeric::array shape,
                    boost::python::numeric::array verts)
    {
    // validate input type and rank
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(ref_angles, PyArray_FLOAT);
    num_util::check_rank(ref_angles, 2);
    num_util::check_type(ref_shape, PyArray_FLOAT);
    num_util::check_rank(ref_shape, 2);
    num_util::check_type(ref_verts, PyArray_FLOAT);
    num_util::check_rank(ref_verts, 2);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(angles, PyArray_FLOAT);
    num_util::check_rank(angles, 2);
    num_util::check_type(shape, PyArray_FLOAT);
    num_util::check_rank(shape, 2);
    num_util::check_type(verts, PyArray_FLOAT);
    num_util::check_rank(verts, 2);
    
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    num_util::check_dim(angles, 1, 1);
    
    num_util::check_dim(shape, 1, 2);
    unsigned int Ns = num_util::shape(shape)[0];
    
    num_util::check_dim(verts, 1, 2);
    unsigned int Nv = num_util::shape(verts)[0];
    
    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];
    
    num_util::check_dim(ref_angles, 1, 1);
    
    num_util::check_dim(ref_shape, 1, 3);
    unsigned int Nref_s = num_util::shape(ref_shape)[0];
    
    num_util::check_dim(ref_verts, 1, 3);
    unsigned int Nref_v = num_util::shape(ref_verts)[0];
    
    // get the raw data pointers and compute the cell list
    float3* ref_points_raw = (float3*) num_util::data(ref_points);
    float* ref_angles_raw = (float*) num_util::data(ref_angles);
    float2* ref_shape_raw = (float2*) num_util::data(ref_shape);
    unsigned int* ref_verts_raw = (unsigned int*) num_util::data(ref_verts);
    float3* points_raw = (float3*) num_util::data(points);
    float* angles_raw = (float*) num_util::data(angles);
    float2* shape_raw = (float2*) num_util::data(shape);
    unsigned int* verts_raw = (unsigned int*) num_util::data(verts);

    compute(ref_points_raw, ref_angles_raw, ref_shape_raw, ref_verts_raw, Nref, Nref_s, Nref_v, points_raw, angles_raw, shape_raw, verts_raw, Np, Ns, Nv);
    }

void export_complement()
    {
    class_<complement>("complement", init<trajectory::Box&, float, float>())
        .def("getBox", &complement::getBox, return_internal_reference<>())
        .def("compute", &complement::computePy)
        .def("getRDF", &complement::getRDFPy)
        .def("getR", &complement::getRPy)
        .def("getNr", &complement::getNrPy)
        //.def("getNmatch", &complement::getNmatchPy)
        ;
    }

}; }; // end namespace freud::complement
