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
    // Causing the rdf to not be run in 2D
    //if (rmax > box.getLz()/2 && !box.is2D())
    //throw invalid_argument("rmax must be smaller than half the smallest box size");

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
//checks out

bool complement::_sameSidePy(boost::python::numeric::array A,
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
    if (dot3(ref, test) >= 0)
        {
        return true;
        }
    else
        {
        return false;
        }
    }

bool complement::_isInsidePy(boost::python::numeric::array t,
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

bool complement::isInside(float2 t[], float2 p)
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

bool complement::isInside(float3 t[], float3 p)
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
    
void complement::_crossPy(boost::python::numeric::array v,
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

float3 complement::cross(float2 v1, float2 v2)
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

// Checks out
float3 complement::cross(float3 v1, float3 v2)
    {
    float3 v;
    v.x = (v1.y * v2.z) - (v2.y * v1.z);
    v.y = (v2.x * v1.z) - (v1.x * v2.z);
    v.z = (v1.x * v2.y) - (v2.x * v1.y);
    return v;
    }

float complement::_dotPy(boost::python::numeric::array v1,
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

float complement::dot2(float2 v1, float2 v2)
    {
    return (v1.x * v2.x) + (v1.y * v2.y);
    }

//Checks out
float complement::dot3(float3 v1, float3 v2)
    {
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    }

//checks out
void complement::_mat_rotPy(boost::python::numeric::array p_rot,
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

float2 complement::mat_rotate(float2 point, float angle)
    {
    float2 rot;
    float mysin = sinf(angle);
    float mycos = cosf(angle);
    rot.x = mycos * point.x - mysin * point.y;
    rot.y = mysin * point.x + mycos * point.y;
    return rot;
    }

void complement::_into_localPy(boost::python::numeric::array local,
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

// btw i need to actually make all the dimensions self-consistent
float2 complement::into_local(float2 ref_point,
                            float2 point,
                            float2 vert,
                            float ref_angle,
                            float angle)
    {
    float2 local;
    local = mat_rotate(mat_rotate(vert, -ref_angle), angle);
    float2 vec;
    // vec.x = (ref_point.x - point.x);
    // vec.y = (ref_point.y - point.y);
    vec.x = point.x - ref_point.x;
    vec.y = point.y - ref_point.y;
    vec = mat_rotate(vec, -ref_angle);
    local.x = local.x + vec.x;
    local.y = local.y + vec.y;
    return local;
    }

float complement::cavity_depth(float2 t[])
    {
        
    // base on cross product
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

void complement::compute(unsigned int* match,
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
                unsigned int Nmaxcheckverts)
    {
    m_nmatch = 0;
    computeWithCellList(match, points, types, angles, shapes, ref_list, check_list, ref_verts, check_verts, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts);
    // if (useCells())
    //     {
    //     // printf("with cells\n");
    //     computeWithCellList(match_raw, points_raw, types_raw, angles_raw, shapes_raw, ref_list_raw, check_list_raw, ref_verts_raw, check_verts_raw, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts);
    //     }
    // else
    //     {
    //     // printf("without cells\n");
    //     computeWithoutCellList(match_raw, points_raw, types_raw, angles_raw, shapes_raw, ref_list_raw, check_list_raw, ref_verts_raw, check_verts_raw, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts);
    //     }
    }

// void complement::computeWithoutCellList(unsigned int* match,
//                 float3* points,
//                 unsigned int* types,
//                 float* angles,
//                 float2* shapes,
//                 unsigned int* ref_list,
//                 unsigned int* check_list,
//                 unsigned int* ref_verts,
//                 unsigned int* check_verts,
//                 unsigned int Np,
//                 unsigned int Nt,
//                 unsigned int Nmaxverts,
//                 unsigned int Nref,
//                 unsigned int Ncheck,
//                 unsigned int Nmaxrefverts,
//                 unsigned int Nmaxcheckverts)
//     {
//     // zero the bin counts for totaling
//     // printf("start\n");
//     memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
//     // printf("memset done\n");
//     float dr_inv = 1.0f / m_dr;
//     // printf("m_dr done\n");
//     float rmaxsq = m_rmax * m_rmax;
//     //unsigned int raw_cnt = 0;
//     #pragma omp parallel
//     {

//     #pragma omp for schedule(guided)
//     // for each reference point
//     for (unsigned int i = 0; i < Nref; i++)
//         {
//         // printf("i\n");
//         for (unsigned int j = 0; j < Np; j++)
//             {
//             // compute r between the two particles
//             // New code here
//             // printf("j\n");
//             for (unsigned int k = 0; k < Nref_v; k++)
//                 {
//                 // printf("k\n");
//                 unsigned int tooth_index = ref_verts[k];
//                 float2 tooth = ref_shape[tooth_index];
//                 // I think this will test against all cavities, not just the one
//                 // If each were on k then it would be different
//                 for (unsigned int l = 0; l < Nv; l++)
//                     {
//                     // printf("l\n");
//                     unsigned int cavity_index = verts[l];
//                     float2 cavity[3];
                    
//                     cavity[0] = shape[cavity_index - 1];
//                     cavity[1] = shape[cavity_index];
//                     cavity[2] = shape[cavity_index + 1];
                    
//                     float depth = cavity_depth(cavity);
                    
//                     for (unsigned int m = 0; m < 3; m++)
//                         {
//                         // printf("m\n");
//                         float2 ref_2D;
//                         float2 point_2D;
//                         ref_2D.x = ref_points[i].x;
//                         ref_2D.y = ref_points[i].y;
//                         point_2D.x = points[j].x;
//                         point_2D.y = points[j].y;
//                         cavity[m] = into_local(ref_2D, point_2D, cavity[m], ref_angles[i], angles[j]);
//                         }
                    
//                     bool test = isInside(cavity, tooth);
                    
//                     if (test == true)
//                         {
//                         //raw_cnt++;
//                         m_nmatch++;
//                         // Calc the relative rdf
//                         float dx = float(ref_points[i].x - points[j].x);
//                         float dy = float(ref_points[i].y - points[j].y);
//                         float dz = float(ref_points[i].z - points[j].z);
                        
//                         float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                        
//                         // Need to scale by cavity depth
                        
//                         float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
//                         if (rsq < rmaxsq)
//                             {
//                             float r = sqrtf(rsq);

//                             // bin that r
//                             float binr = r * dr_inv / depth;
//                             // fast float to int conversion with truncation
//                             #ifdef __SSE2__
//                                 unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
//                             #else
//                                 unsigned int bin = (unsigned int)(binr);
//                             #endif
//                             #pragma omp atomic
//                                 m_bin_counts[bin]++;
//                             }
//                         }
//                     }
//                 }
//             }
//         } // done looping over reference points
//     } // End of parallel section

//     // now compute the "rdf"
//     float ndens = float(Np) / m_box.getVolume();
//     m_rdf_array[0] = 0.0f;
//     m_N_r_array[0] = 0.0f;
//     m_N_r_array[1] = 0.0f;

//     for (unsigned int bin = 1; bin < m_nbins; bin++)
//         {
//         float avg_counts = m_bin_counts[bin] / float(Nref);
//         m_rdf_array[bin] = avg_counts / m_vol_array[bin] / ndens;

//         if (bin+1 < m_nbins)
//             m_N_r_array[bin+1] = m_N_r_array[bin] + avg_counts;
//         }
//     }

void complement::computeWithCellList(unsigned int* match,
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
                unsigned int Nmaxcheckverts)
    {
    // assert(ref_points);
    // assert(points);
    // assert(Nref > 0);
    // assert(Np > 0);
    m_nP = Np;
    printf("%i\n", Nref);
    // Is it not getting the other particle
    
    
    m_lc->computeCellList(points, Np);
    
    // zero the bin counts for totaling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    // printf("I bet this is the seg fault\n");
    // m_match_array = boost::shared_array<unsigned int>(new unsigned int[m_nP]);
    // memset((void*)m_match_array.get(), 0, sizeof(unsigned int)*m_nP);
    // printf("did it seg fault\n");
    float dr_inv = 1.0f / m_dr;
    // I feel like this should be calculated here, or rather we know that it is 1 because it is normallized
    float rmaxsq = m_rmax * m_rmax;
    //unsigned int raw_cnt = 0;
    #pragma omp parallel
    {
    
    #pragma omp for schedule(guided)
    // for each reference point
    for (unsigned int i = 0; i < Np; i++)
        {
        // match[i] = 0;
        // need to set up the mask
        float3 point = points[i];
        unsigned int type = types[i];
        // find if type in refs
        bool in_refs = false;
        for (unsigned int ref_idx = 0; ref_idx < Nref; ref_idx++)
            {
            if (ref_list[ref_idx] == type)
                {
                in_refs = true;
                }
            }
        if (in_refs != true)
            {
            continue;
            }
        // get the cell the point is in
        // printf("particle %i\n", i);
        unsigned int ref_cell = m_lc->getCell(point);
        
        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            // printf("neighbor %i\n", neigh_idx);
            unsigned int neigh_cell = neigh_cells[neigh_idx];
            
            // iterate over the particles in that cell
            locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                {
                printf("i = %i j = %i\n", i, j);
                
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
                
                // I bet that I don't need this now that it won't check the bad ones...
                if (i == j)
                    {
                    printf("%i==%i...\n", i, j);
                    continue;
                    }
                
                // iterate over the verts in the ref particle
                for (unsigned int k = 0; k < Nmaxrefverts; k++)
                    {
                    // I do believe there will be an issue if diff amount of teeth
                    printf("tooth/cavity = %i\n", k);
                    unsigned int tooth_index = ref_verts[k];
                    // This may not be the greatest way to get the shape...or I may need another function
                    // float2 shape = shapes[type];
                    float2 tooth = shapes[type * Nmaxverts + tooth_index];
                    
                    // This would be for all of the cavities, just want the matching one
                    // for (unsigned int l = 0; l < Nv; l++)
                    //     {
                        // printf("cavity %i\n", l);
                        // unsigned int cavity_index = verts[l];
                    unsigned int cavity_index = check_verts[k];
                    float2 cavity [3];
                
                    cavity[0] = shapes[check_type * Nmaxverts + cavity_index - 1];
                    cavity[1] = shapes[check_type * Nmaxverts + cavity_index];
                    cavity[2] = shapes[check_type * Nmaxverts + cavity_index + 1];
                    
                    float depth = cavity_depth(cavity);
                    
                    for (unsigned int m = 0; m < 3; m++)
                        {
                        float2 ref_2D;
                        float2 point_2D;
                        ref_2D.x = point.x;
                        ref_2D.y = point.y;
                        point_2D.x = check.x;
                        point_2D.y = check.y;
                        cavity[m] = into_local(ref_2D, point_2D, cavity[m], angles[i], angles[j]);
                        printf("%f %f\n", cavity[m].x, cavity[m].y);
                        }
                        
                        // printf("%f %f %f %f %f %f\n", cavity[0].x, cavity[0].y, cavity[1].x, cavity[1].y, cavity[2].x, cavity[2].y);
                        // printf("testing if isInside\n");
                        // return list of matching particles
                        // This sounds like it isn't working
                        // bool test = isInside(cavity, tooth);
                        // float3 m_cav [3];
                        // for (int c_idx = 0; c_idx < 3; c_idx++)
                        //     {
                        //     m_cav[c_idx].x = cavity[c_idx].x;
                        //     m_cav[c_idx].y = cavity[c_idx].y;
                        //     m_cav[c_idx].z = 0;
                        //     }
                        // float3 m_t;
                        // m_t.x = tooth.x;
                        // m_t.y = tooth.y;
                        // m_t.z = 0.0;
                        // // No guarantee :(
                        // bool test = sameSide(m_cav[0], m_cav[2], m_cav[1], m_t);
                        
                        if (isInside(cavity, tooth))
                            {
                            printf("particle %i is inside particle %i\n", j, i);
                            match[i] = 1;
                            match[j] = 1;
                            // printf("shit was inside\n");
                            //raw_cnt++;
                            // printf("value of match array[%i] = %i\n", i, m_match_array[i]);
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
                                // float binr = r * dr_inv / depth;
                            float binr = r * dr_inv;
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
                                // printf("bin %i\n", bin);
                                m_bin_counts[bin]++;
                                // }
                        
                            }
                        
                        // }
                    
                    }
                }
            }
        } // done looping over reference points
    } // End of parallel section
    // printf("starting to compute rdf\n");
    // now compute the rdf
    // Um most of mine might be in the first bin...
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

void complement::computePy(boost::python::numeric::array match,
                    boost::python::numeric::array points,
                    boost::python::numeric::array types,
                    boost::python::numeric::array angles,
                    boost::python::numeric::array shapes,
                    boost::python::numeric::array ref_list,
                    boost::python::numeric::array check_list,
                    boost::python::numeric::array ref_verts,
                    boost::python::numeric::array check_verts)
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
    // printf("done with rank checks\n");
    // validate that the 2nd dimension is only 3
    
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    num_util::check_dim(types, 0, Np);
    num_util::check_dim(angles, 0, Np);
    
    num_util::check_dim(shapes, 2, 2);
    unsigned int Nt = num_util::shape(shapes)[0];
    unsigned int Nmaxverts = num_util::shape(shapes)[1];
    
    unsigned int Nref = num_util::shape(ref_list)[0];
    unsigned int Ncheck = num_util::shape(check_list)[0];
    
    num_util::check_dim(ref_verts, 0, Nref);
    num_util::check_dim(check_verts, 0, Ncheck);
    
    unsigned int Nmaxrefverts = num_util::shape(ref_verts)[0];
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

    compute(match_raw, points_raw, types_raw, angles_raw, shapes_raw, ref_list_raw, check_list_raw, ref_verts_raw, check_verts_raw, Np, Nt, Nmaxverts, Nref, Ncheck, Nmaxrefverts, Nmaxcheckverts);
    }

void export_complement()
    {
    class_<complement>("complement", init<trajectory::Box&, float, float>())
        .def("getBox", &complement::getBox, return_internal_reference<>())
        .def("compute", &complement::computePy)
        .def("getRDF", &complement::getRDFPy)
        .def("getR", &complement::getRPy)
        .def("getNr", &complement::getNrPy)
        .def("getNpair", &complement::getNpairPy)
        .def("_sameSide", &complement::_sameSidePy)
        .def("_isInside", &complement::_isInsidePy)
        .def("_cross", &complement::_crossPy)
        .def("_dot3", &complement::_dotPy)
        .def("_mat_rot", &complement::_mat_rotPy)
        .def("_into_local", &complement::_into_localPy)
        //.def("getNmatch", &complement::getNmatchPy)
        ;
    }

}; }; // end namespace freud::complement
