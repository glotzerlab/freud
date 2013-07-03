#include "pair.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost::python;

namespace freud { namespace pair {

pair::pair(const trajectory::Box& box, float rmax, float dr)
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
    // Running in the pair branch should force the box to be 2d...at least if a pos file
    // Unsure if the same for hacked dcd
    // if (rmax > box.getLz()/2 && !box.is2D())
    // throw invalid_argument("rmax must be smaller than half the smallest box size");

    m_nbins = int(floorf(m_rmax / m_dr));
    assert(m_nbins > 0);
    m_SoM_array = boost::shared_array<float>(new float[m_nbins]);
    memset((void*)m_SoM_array.get(), 0, sizeof(float)*m_nbins);
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

pair::~pair()
    {
    if(useCells())
    delete m_lc;
    }

bool pair::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());
    if (m_box.is2D())
    l_min = fmin(l_min, m_box.getLy());
    if (m_rmax < l_min/3)
    return true;
    return false;
    }

bool pair::_sameSidePy(boost::python::numeric::array A,
                            boost::python::numeric::array B,
                            boost::python::numeric::array r,
                            boost::python::numeric::array p)
    {
    num_util::check_type(A, PyArray_FLOAT);
    num_util::check_rank(A, 1);
    num_util::check_dim(A, 0, 3);
    num_util::check_type(B, PyArray_FLOAT);
    num_util::check_rank(B, 1);
    num_util::check_dim(B, 0, 3);
    num_util::check_type(r, PyArray_FLOAT);
    num_util::check_rank(r, 1);
    num_util::check_dim(r, 0, 3);
    num_util::check_type(p, PyArray_FLOAT);
    num_util::check_rank(p, 1);
    num_util::check_dim(p, 0, 3);
    
    float3* A_raw = (float3*) num_util::data(A);
    float3* B_raw = (float3*) num_util::data(B);
    float3* r_raw = (float3*) num_util::data(r);
    float3* p_raw = (float3*) num_util::data(p);
    
    return sameSide(*A_raw, *B_raw, *r_raw, *p_raw);
    }

// Need to cite this
bool pair::sameSide(float3 A, float3 B, float3 r, float3 p)
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
    if (dot3(ref, test) >= 0)
        {
        return true;
        }
    else
        {
        return false;
        }
    }

bool pair::_isInsidePy(boost::python::numeric::array t,
                            boost::python::numeric::array p)
    {
    num_util::check_type(t, PyArray_FLOAT);
    num_util::check_rank(t, 2);
    num_util::check_dim(t, 0, 3);
    num_util::check_dim(t, 1, 2);
    num_util::check_type(p, PyArray_FLOAT);
    num_util::check_rank(p, 1);
    num_util::check_dim(p, 0, 2);
    
    float3* t_raw = (float3*) num_util::data(t);
    
    float3* p_raw = (float3*) num_util::data(p);
    
    return isInside(t_raw, *p_raw);
    }

bool pair::isInside(float2 t[], float2 p)
    {
    float3 nt [3];
    float3 np;
    
    for (unsigned int i = 0; i < 3; i++)
        {
        nt[i].x = t[i].x;
        nt[i].y = t[i].y;
        nt[i].z = 0;
        }
    
    np.x = p.x;
    np.y = p.y;
    np.z = 0;
    
    return isInside(nt, np);
      
    }

bool pair::isInside(float3 t[], float3 p)
    {
    float3 A;
    float3 B;
    float3 C;
    float3 P;
    
    // Even though float threes are taken in, the z component is assumed zero
    // i.e. all in the same plane
    
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
    
void pair::_crossPy(boost::python::numeric::array v,
                        boost::python::numeric::array v1,
                        boost::python::numeric::array v2)
    {
    num_util::check_type(v, PyArray_FLOAT);
    num_util::check_rank(v, 1);
    num_util::check_dim(v, 0, 3);
    num_util::check_type(v1, PyArray_FLOAT);
    num_util::check_rank(v1, 1);
    num_util::check_dim(v1, 0, 3);
    num_util::check_type(v2, PyArray_FLOAT);
    num_util::check_rank(v2, 1);
    num_util::check_dim(v2, 0, 3);
    
    float3* v_raw = (float3*) num_util::data(v);
    float3* v1_raw = (float3*) num_util::data(v1);
    float3* v2_raw = (float3*) num_util::data(v2);
    *v_raw = cross(*v1_raw, *v2_raw);
    }

float3 pair::cross(float2 v1, float2 v2)
    {
    float3 v1_n;
    float3 v2_n;
    v1_n.x = v1.x;
    v1_n.y = v1.y;
    v1_n.z = 0;
    v2_n.x = v2.x;
    v2_n.y = v2.y;
    v2_n.z = 0;
    return cross(v1_n, v2_n);
    }

float3 pair::cross(float3 v1, float3 v2)
    {
    float3 v;
    v.x = (v1.y * v2.z) - (v2.y * v1.z);
    v.y = (v2.x * v1.z) - (v1.x * v2.z);
    v.z = (v1.x * v2.y) - (v2.x * v1.y);
    return v;
    }

float pair::_dotPy(boost::python::numeric::array v1,
                        boost::python::numeric::array v2)
    {
    num_util::check_type(v1, PyArray_FLOAT);
    num_util::check_rank(v1, 1);
    num_util::check_dim(v1, 0, 3);
    num_util::check_type(v2, PyArray_FLOAT);
    num_util::check_rank(v2, 1);
    num_util::check_dim(v2, 0, 3);
    
    float3* v1_raw = (float3*) num_util::data(v1);
    float3* v2_raw = (float3*) num_util::data(v2);
    return dot3(*v1_raw, *v2_raw);
    }

float pair::dot2(float2 v1, float2 v2)
    {
    return (v1.x * v2.x) + (v1.y * v2.y);
    }

float pair::dot3(float3 v1, float3 v2)
    {
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    }

void pair::_mat_rotPy(boost::python::numeric::array p_rot,
                        boost::python::numeric::array p,
                        float angle)
    {
    
    num_util::check_type(p_rot, PyArray_FLOAT);
    num_util::check_rank(p_rot, 1);
    num_util::check_dim(p_rot, 0, 2);
    
    num_util::check_type(p, PyArray_FLOAT);
    num_util::check_rank(p, 1);
    num_util::check_dim(p, 0, 2);
    
    float2* p_rot_raw = (float2*) num_util::data(p_rot);
    float2* p_raw = (float2*) num_util::data(p);
    *p_rot_raw = mat_rotate(*p_raw, angle);
    }

float2 pair::mat_rotate(float2 point, float angle)
    {
    float2 rot;
    float mysin = sinf(angle);
    float mycos = cosf(angle);
    rot.x = mycos * point.x - mysin * point.y;
    rot.y = mysin * point.x + mycos * point.y;
    return rot;
    }

void pair::_into_localPy(boost::python::numeric::array local,
                        boost::python::numeric::array p_ref,
                        boost::python::numeric::array p,
                        boost::python::numeric::array vert,
                        float a_ref,
                        float a)
    {
    
    num_util::check_type(local, PyArray_FLOAT);
    num_util::check_rank(local, 1);
    num_util::check_dim(local, 0, 2);
    
    num_util::check_type(p_ref, PyArray_FLOAT);
    num_util::check_rank(p_ref, 1);
    num_util::check_dim(p_ref, 0, 2);
    
    num_util::check_type(p, PyArray_FLOAT);
    num_util::check_rank(p, 1);
    num_util::check_dim(p, 0, 2);
    
    num_util::check_type(vert, PyArray_FLOAT);
    num_util::check_rank(vert, 1);
    num_util::check_dim(vert, 0, 2);
    
    float2* local_raw = (float2*) num_util::data(local);
    float2* p_ref_raw = (float2*) num_util::data(p_ref);
    float2* p_raw = (float2*) num_util::data(p);
    float2* vert_raw = (float2*) num_util::data(vert);
    *local_raw = into_local(*p_ref_raw, *p_raw, *vert_raw, a_ref, a);
    }

float2 pair::into_local(float2 ref_point,
                            float2 point,
                            float2 vert,
                            float ref_angle,
                            float angle)
    {
    float2 local;
    local = mat_rotate(mat_rotate(vert, -ref_angle), angle);
    float2 vec;
    vec.x = point.x - ref_point.x;
    vec.y = point.y - ref_point.y;
    vec = mat_rotate(vec, -ref_angle);
    local.x = local.x + vec.x;
    local.y = local.y + vec.y;
    return local;
    }

float pair::cavity_depth(float2 t[])
    {
        
    float2 v_mouth;
    float2 v_side;
    
    v_mouth.x = t[0].x - t[2].x;
    v_mouth.y = t[0].y - t[2].y;
    float m_mouth = sqrt(dot2(v_mouth, v_mouth));
    v_side.x = t[1].x - t[2].x;
    v_side.y = t[1].y - t[2].y;
    
    float3 a_vec = cross(v_mouth, v_side);
    float area = sqrt(dot3(a_vec, a_vec));
    return area/m_mouth;
    }

void pair::compute(unsigned int* match,
                float3* points,
                unsigned int* types,
                float* angles,
                float2* shapes,
                unsigned int* ref_list,
                unsigned int* check_list,
                unsigned int* ref_verts,
                unsigned int* check_verts,
                unsigned int Np,
                unsigned int Nt,
                unsigned int Nmaxverts,
                unsigned int Nref,
                unsigned int Ncheck,
                unsigned int Nmaxrefverts,
                unsigned int Nmaxcheckverts,
                float depth)
    {
    m_nmatch = 0;
    // computeWithCellList(match, points, types, angles, shapes, ref_list, check_list, ref_verts, check_verts, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts);
    if (useCells())
        {
        // printf("with cells\n");
        computeWithCellList(match, points, types, angles, shapes, ref_list, check_list, ref_verts, check_verts, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts, depth);
        }
    else
        {
        // printf("without cells\n");
        computeWithoutCellList(match, points, types, angles, shapes, ref_list, check_list, ref_verts, check_verts, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts, depth);
        }
    }

void pair::computeWithoutCellList(unsigned int* match,
                float3* points,
                unsigned int* types,
                float* angles,
                float2* shapes,
                unsigned int* ref_list,
                unsigned int* check_list,
                unsigned int* ref_verts,
                unsigned int* check_verts,
                unsigned int Np,
                unsigned int Nt,
                unsigned int Nmaxverts,
                unsigned int Nref,
                unsigned int Ncheck,
                unsigned int Nmaxrefverts,
                unsigned int Nmaxcheckverts,
                float depth)
    {
    m_nP = Np;
    // zero the bin counts for totaling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;
    #pragma omp parallel
    {
    #pragma omp for schedule(guided)
    // for each reference point
    for (unsigned int i = 0; i < m_nP; i++)
        {
        // grab point and type
        // might be eliminated later for efficiency
        float3 point = points[i];
        unsigned int ref_type = types[i];
        // find if type in refs
        bool in_refs = false;
        for (unsigned int ref_idx = 0; ref_idx < Nref; ref_idx++)
            {
            if (ref_list[ref_idx] == ref_type)
                {
                in_refs = true;
                }
            }
        if (in_refs != true)
            {
            continue;
            }
        for (unsigned int j = 0; j < m_nP; j++)
            {
            float3 check = points[j];
            unsigned int check_type = types[j];
            // find if type in refs
            bool in_check = false;
            for (unsigned int check_idx = 0; check_idx < Ncheck; check_idx++)
                {
                if (check_list[check_idx] == check_type)
                    {
                    in_check = true;
                    }
                }
            if (in_check != true)
                {
                continue;
                }
            // retained for self-pairary particles
            // will skip same particle
            if (i == j)
                {
                continue;
                }
            // iterate over the verts in the ref particle
            for (unsigned int k = 0; k < Nmaxrefverts; k++)
                {
                unsigned int tooth_index = ref_verts[k];
                float2 tooth = shapes[ref_type * Nmaxverts + tooth_index];
                unsigned int cavity_index = check_verts[k];
                float2 cavity [3];
                cavity[0] = shapes[check_type * Nmaxverts + cavity_index - 1];
                cavity[1] = shapes[check_type * Nmaxverts + cavity_index];
                cavity[2] = shapes[check_type * Nmaxverts + cavity_index + 1];
                    
                for (unsigned int m = 0; m < 3; m++)
                    {
                    float2 ref_2D;
                    float2 point_2D;
                    ref_2D.x = point.x;
                    ref_2D.y = point.y;
                    point_2D.x = check.x;
                    point_2D.y = check.y;
                    cavity[m] = into_local(ref_2D, point_2D, cavity[m], angles[i], angles[j]);
                    }
                
                if (isInside(cavity, tooth))
                    {
                    m_nmatch++;
                    // Calc the relative rdf
                    float dx = float(point.x - check.x);
                    float dy = float(point.y - check.y);
                    float dz = float(point.z - check.z);
                    
                    float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                    
                    float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    float r = sqrtf(rsq);
                    // bin that r
                    float binr = r * dr_inv; // / depth;
                    // fast float to int conversion with truncation
                    // #ifdef __SSE2__
                    //     unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                    // #else
                    //     unsigned int bin = (unsigned int)(binr);
                    // #endif
                    unsigned int bin = (unsigned int)(binr);
                    #pragma omp atomic
                        m_bin_counts[bin]++;
                    }
                }
            }
        } // done looping over reference points
    } // End of parallel section

    // now compute SoM
    float ndens = float(m_nP) / m_box.getVolume();
    m_SoM_array[0] = 0.0f;
    m_N_r_array[0] = 0.0f;
    m_N_r_array[1] = 0.0f;

    for (unsigned int bin = 0; bin < m_nbins; bin++)
        {
        float avg_counts = m_bin_counts[bin] / float(m_nP);
        m_SoM_array[bin] = avg_counts / m_vol_array[bin] / ndens;

        if (bin+1 < m_nbins)
            m_N_r_array[bin+1] = m_N_r_array[bin] + avg_counts;
        }
    }

void pair::computeWithCellList(unsigned int* match,
                float3* points,
                unsigned int* types,
                float* angles,
                float2* shapes,
                unsigned int* ref_list,
                unsigned int* check_list,
                unsigned int* ref_verts,
                unsigned int* check_verts,
                unsigned int Np,
                unsigned int Nt,
                unsigned int Nmaxverts,
                unsigned int Nref,
                unsigned int Ncheck,
                unsigned int Nmaxrefverts,
                unsigned int Nmaxcheckverts,
                float depth)
    {
    m_nP = Np;
    m_lc->computeCellList(points, m_nP);
    
    // zero the bin counts for totaling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;
    #pragma omp parallel
    {
    #pragma omp for schedule(guided)
    // for each particle
    for (unsigned int i = 0; i < m_nP; i++)
        {
        // grab point and type
        // might be eliminated later for efficiency
        float3 point = points[i];
        unsigned int ref_type = types[i];
        // find if type in refs
        bool in_refs = false;
        for (unsigned int ref_idx = 0; ref_idx < Nref; ref_idx++)
            {
            if (ref_list[ref_idx] == ref_type)
                {
                in_refs = true;
                }
            }
        if (in_refs != true)
            {
            continue;
            }
        // get the cell the point is in
        unsigned int ref_cell = m_lc->getCell(point);
        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];
            // iterate over the particles in that cell
            locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                {
                float3 check = points[j];
                unsigned int check_type = types[j];
                // find if type in refs
                bool in_check = false;
                for (unsigned int check_idx = 0; check_idx < Ncheck; check_idx++)
                    {
                    if (check_list[check_idx] == check_type)
                        {
                        in_check = true;
                        }
                    }
                if (in_check != true)
                    {
                    continue;
                    }
                // retained for self-pairary particles
                // will skip same particle
                if (i == j)
                    {
                    continue;
                    }
                // iterate over the verts in the ref particle
                for (unsigned int k = 0; k < Nmaxrefverts; k++)
                    {
                    // requires that the size of the two arrays are the same
                    // ok because it's supposed to be perfect match
                    unsigned int tooth_index = ref_verts[k];
                    float2 tooth = shapes[ref_type * Nmaxverts + tooth_index];
                    // allows for different vertex'd shapes to run with dummies at the end
                    // if (tooth.x == nan)
                    //     {
                    //     continue;
                    //     }
                    unsigned int cavity_index = check_verts[k];
                    float2 cavity [3];
                    cavity[0] = shapes[check_type * Nmaxverts + cavity_index - 1];
                    cavity[1] = shapes[check_type * Nmaxverts + cavity_index];
                    cavity[2] = shapes[check_type * Nmaxverts + cavity_index + 1];
                    
                    for (unsigned int m = 0; m < 3; m++)
                        {
                        float2 ref_2D;
                        float2 point_2D;
                        ref_2D.x = point.x;
                        ref_2D.y = point.y;
                        point_2D.x = check.x;
                        point_2D.y = check.y;
                        cavity[m] = into_local(ref_2D, point_2D, cavity[m], angles[i], angles[j]);
                        }
                        
                        if (isInside(cavity, tooth))
                            {
                            match[i] = 1;
                            match[j] = 1;
                            m_nmatch++;
                            // Calc the relative rdf
                            float dx = float(point.x - check.x);
                            float dy = float(point.y - check.y);
                            float dz = float(point.z - check.z);
                        
                            float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                            float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        
                            // if (rsq < rmaxsq)
                                // {
                            float r = sqrtf(rsq);
                                // This selects the bin
                            float binr = r * dr_inv; // / depth;
                            // float binr = r * dr_inv;
                                // printf("binr %f\n", binr);
                                // fast float to int conversion with truncation
                                // #ifdef __SSE2__
                                //     printf("SSE2\n");
                                //     unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                                // #else
                                //     printf("nonSSE\n");
                                //     unsigned int bin = (unsigned int)(binr);
                                // #endif
                                
                                // Bug was there, not sure why/what
                                
                                
                            unsigned int bin = (unsigned int)(binr);
                            #pragma omp atomic
                                m_bin_counts[bin]++;
                        
                            }
                    
                        }
                    }
                }
            } // done looping over reference points
        } // End of parallel section
    // printf("starting to compute rdf\n");
    // now compute the rdf
    // Um most of mine might be in the first bin...
    float ndens = float(m_nP) / m_box.getVolume();
    m_SoM_array[0] = 0.0f;
    m_N_r_array[0] = 0.0f;
    m_N_r_array[1] = 0.0f;
    
    // This might not be working right as it zeros the first bin...
    for (unsigned int bin = 0; bin < m_nbins; bin++)
        {
        float avg_counts = m_bin_counts[bin] / float(m_nP);
        m_SoM_array[bin] = avg_counts; // / m_vol_array[bin] / ndens;
        
        if (bin+1 < m_nbins)
            {
            m_N_r_array[bin+1] = m_N_r_array[bin] + avg_counts;
            }
        }
    }

void pair::computePy(boost::python::numeric::array match,
                    boost::python::numeric::array points,
                    boost::python::numeric::array types,
                    boost::python::numeric::array angles,
                    boost::python::numeric::array shapes,
                    boost::python::numeric::array ref_list,
                    boost::python::numeric::array check_list,
                    boost::python::numeric::array ref_verts,
                    boost::python::numeric::array check_verts,
                    float depth)
    {
    // points contains all the particle positions; Np x 3
    // types contains all the types; Np (x 1)
    // angles contains the angle of each particle; Np (x1)
    // shapes contains the verticies of each type; Nt x Nv_max
    //      where any unpop'd vertices are nan
    // ref_list contains the types of shapes that will be referenced
    // check_list contains the types of shapes that will be checked
    // ref_verts contains the vert index that will be checked
    // check_verts contains the vert index that will be checked
    // match will contain the particle index of the ref that is matched
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(types, PyArray_INT);
    num_util::check_rank(types, 1);
    num_util::check_type(angles, PyArray_FLOAT);
    num_util::check_rank(angles, 1);
    num_util::check_type(shapes, PyArray_FLOAT);
    num_util::check_rank(shapes, 3);
    num_util::check_type(ref_list, PyArray_INT);
    num_util::check_rank(ref_list, 1);
    num_util::check_type(check_list, PyArray_INT);
    num_util::check_rank(check_list, 1);
    num_util::check_type(ref_verts, PyArray_INT);
    num_util::check_rank(ref_verts, 1);
    num_util::check_type(check_verts, PyArray_INT);
    num_util::check_rank(check_verts, 1);
    num_util::check_type(match, PyArray_INT);
    num_util::check_rank(match, 1);
    
    // get the number of particles
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    //validate that the types and angles coming in are the correct size
    num_util::check_dim(types, 0, Np);
    num_util::check_dim(angles, 0, Np);
    
    // validate that the shapes array is 3d
    num_util::check_dim(shapes, 2, 2);
    // establish the number of types
    unsigned int Nt = num_util::shape(shapes)[0];
    // establish the max number of verts
    unsigned int Nmaxverts = num_util::shape(shapes)[1];
    
    // establish the number of reference and check particles
    unsigned int Nref = num_util::shape(ref_list)[0];
    unsigned int Ncheck = num_util::shape(check_list)[0];
    
    // This isn't quite right
    // num_util::check_dim(ref_verts, 0, Nref);
    // num_util::check_dim(check_verts, 0, Ncheck);
    
    unsigned int Nmaxrefverts = num_util::shape(ref_verts)[0];
    // This is expressly for the same number of cavities
    num_util::check_dim(check_verts, 0, Nmaxrefverts);
    unsigned int Nmaxcheckverts = num_util::shape(check_verts)[0];
    
    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);
    unsigned int* types_raw = (unsigned int*) num_util::data(types);
    float* angles_raw = (float*) num_util::data(angles);
    float2* shapes_raw = (float2*) num_util::data(shapes);
    unsigned int* ref_list_raw = (unsigned int*) num_util::data(ref_list);
    unsigned int* check_list_raw = (unsigned int*) num_util::data(check_list);
    unsigned int* ref_verts_raw = (unsigned int*) num_util::data(ref_verts);
    unsigned int* check_verts_raw = (unsigned int*) num_util::data(check_verts);
    unsigned int* match_raw = (unsigned int*) num_util::data(match);

    compute(match_raw, points_raw, types_raw, angles_raw, shapes_raw, ref_list_raw, check_list_raw, ref_verts_raw, check_verts_raw, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts, depth);
    }

void export_pair()
    {
    class_<pair>("pair", init<trajectory::Box&, float, float>())
        .def("getBox", &pair::getBox, return_internal_reference<>())
        .def("compute", &pair::computePy)
        .def("getSoM", &pair::getSoMPy)
        .def("getR", &pair::getRPy)
        .def("getNr", &pair::getNrPy)
        .def("getNpair", &pair::getNpairPy)
        .def("_sameSide", &pair::_sameSidePy)
        .def("_isInside", &pair::_isInsidePy)
        .def("_cross", &pair::_crossPy)
        .def("_dot3", &pair::_dotPy)
        .def("_mat_rot", &pair::_mat_rotPy)
        .def("_into_local", &pair::_into_localPy)
        ;
    }

}; }; // end namespace freud::pair
