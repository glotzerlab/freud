#include "pairing.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;

using namespace tbb;

namespace freud { namespace pairing {

inline bool comp_check_2D(const float rmax,
                          const trajectory::Box& box,
                          const float3 r_i,
                          const float3 r_j,
                          const float angle_s_i,
                          const float angle_s_j,
                          const float angle_c_i,
                          const float angle_c_j,
                          const float shape_dot_target,
                          const float shape_dot_tol,
                          const float comp_dot_target,
                          const float comp_dot_tol,
                          float& dist2,
                          float& sdot,
                          float& cdot)
    {
    float rmaxsq = rmax * rmax;
    float2 r_ij;
    float2 r_ij_u;
    float2 r_ji_u;
    float dx = r_j.x - r_i.x;
    float dy = r_j.y - r_i.y;
    float dz = (float) 0.0;
    float3 delta = box.wrap(make_float3(dx, dy, dz));
    r_ij.x = delta.x;
    r_ij.y = delta.y;
    float r_ij_mag = sqrt(dot2(r_ij, r_ij));
    r_ij_u.x = r_ij.x / r_ij_mag;
    r_ij_u.y = r_ij.y / r_ij_mag;
    r_ji_u.x = -r_ij_u.x;
    r_ji_u.y = -r_ij_u.y;
    float2 theta_s_i;
    theta_s_i.x = cos(angle_s_i);
    theta_s_i.y = sin(angle_s_i);
    float2 theta_s_j;
    theta_s_j.x = cos(angle_s_j);
    theta_s_j.y = sin(angle_s_j);
    float2 theta_c_i;
    theta_c_i.x = cos(angle_c_i);
    theta_c_i.y = sin(angle_c_i);
    float2 theta_c_j;
    theta_c_j.x = cos(angle_c_j);
    theta_c_j.y = sin(angle_c_j);
    float d_ij = dot2(r_ij, r_ij);
    float theta_s_ij = dot2(theta_s_i, theta_s_j);
    float theta_c_ij = dot2(theta_c_i, theta_c_j);
    float v_ij = dot2(r_ij_u, theta_c_i);
    float v_ji = dot2(r_ji_u, theta_c_j);
    dist2 = d_ij;
    sdot = theta_s_ij;
    cdot = v_ij;
    // printf("%f\n", sdot);
    // printf("%f\n", cdot);
    // determine if paired
    if (d_ij > rmaxsq)
        return false;
    // if ((theta_s_ij < (shape_dot_target - shape_dot_tol)) || (theta_s_ij > (shape_dot_target + shape_dot_tol)))
    //     return false;
    // if ((v_ij < (comp_dot_target - comp_dot_tol)) || (v_ij > (comp_dot_target + comp_dot_tol)))
    //     return false;
    // if ((v_ji < (comp_dot_target - comp_dot_tol)) || (v_ji > (comp_dot_target + comp_dot_tol)))
    //     return false;
    if (abs(theta_s_ij - shape_dot_target) > shape_dot_tol)
        return false;
    if (abs(theta_c_ij - comp_dot_target) > comp_dot_tol)
        return false;
    if (v_ij < 0)
        return false;
    if (v_ji < 0)
        return false;
    return true;
    }

pairing::pairing(const trajectory::Box& box,
                 const float rmax,
                 const unsigned int k,
                 const float comp_dot_target,
                 const float comp_dot_tol)
    : m_box(box), m_rmax(rmax), m_k(k), m_comp_dot_target(comp_dot_target), m_comp_dot_tol(comp_dot_tol)
    {
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (rmax > box.getLx()/2 || rmax > box.getLy()/2)
    throw invalid_argument("rmax must be smaller than half the smallest box size");
    if (rmax > box.getLz()/2 && !box.is2D())
        throw invalid_argument("rmax must be smaller than half the smallest box size");
    if (useCells())
        {
        m_nn = new locality::LinkCell(box, rmax);
        }
    }

pairing::~pairing()
    {
    if(useCells())
    delete m_lc;
    }

class ComputePairing2DCellList
    {
    private:
        unsigned int *m_match_array;
        float* m_dist2_array;
        float* m_sdot_array;
        float* m_cdot_array;
        const float3 *m_points;
        const float *m_shape_angles;
        const float *m_comp_angles;
        const unsigned int m_Np;
        const locality::LinkCell* m_lc;
        const trajectory::Box m_box;
        const float m_rmax;
        const float m_shape_dot_target;
        const float m_shape_dot_tol;
        const float m_comp_dot_target;
        const float m_comp_dot_tol;

    public:
        ComputePairing2DCellList(unsigned int *match_array,
                                 float* dist2_array,
                                 float* sdot_array,
                                 float* cdot_array,
                                 const float3 *points,
                                 const float *shape_angles,
                                 const float *comp_angles,
                                 const unsigned int Np,
                                 const locality::LinkCell* lc,
                                 const float r_max,
                                 const trajectory::Box box,
                                 const float shape_dot_target,
                                 const float shape_dot_tol,
                                 const float comp_dot_target,
                                 const float comp_dot_tol)
        : m_match_array(match_array), m_dist2_array(dist2_array), m_sdot_array(sdot_array), m_cdot_array(cdot_array),
          m_points(points), m_shape_angles(shape_angles), m_comp_angles(comp_angles), m_Np(Np), m_lc(lc), m_rmax(r_max),
          m_box(box), m_shape_dot_target(shape_dot_target), m_shape_dot_tol(shape_dot_tol),
          m_comp_dot_target(comp_dot_target), m_comp_dot_tol(comp_dot_tol)
        {
        }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // m_nP = Np;
            // for each particle
            float dist2;
            float sdot;
            float cdot;
            dist2 = 0.0;
            sdot = 0.0;
            cdot = 0.0;
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                float3 r_i = m_points[i];
                float angle_s_i = m_shape_angles[i];
                float angle_c_i = m_comp_angles[i];
                // get the cell the point is in
                unsigned int ref_cell = m_lc->getCell(r_i);
                // loop over all neighboring cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    unsigned int neigh_cell = neigh_cells[neigh_idx];
                    // iterate over the particles in that cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                        {
                        float3 r_j = m_points[j];
                        float angle_s_j = m_shape_angles[j];
                        float angle_c_j = m_comp_angles[j];
                        // will skip same particle
                        if (i == j)
                            {
                            continue;
                            }

                        if (comp_check_2D(m_rmax,
                                       m_box,
                                       r_i,
                                       r_j,
                                       angle_s_i,
                                       angle_s_j,
                                       angle_c_i,
                                       angle_c_j,
                                       m_shape_dot_target,
                                       m_shape_dot_tol,
                                       m_comp_dot_target,
                                       m_comp_dot_tol,
                                       dist2,
                                       sdot,
                                       cdot))
                            {
                            m_match_array[i] = 1;
                            // m_match_array[j] = 1;
                            m_dist2_array[i] = dist2;
                            // m_dist2_array[j] = dist2;
                            m_sdot_array[i] = sdot;
                            // m_sdot_array[j] = sdot;
                            m_cdot_array[i] = cdot;
                            // m_cdot_array[j] = cdot;
                            }
                        } // done looping over neighbors
                    } // done looping over neighbor cells
                } // done looping over reference points
            }
    };

class ComputePairing2DWithoutCellList
    {
    private:
        unsigned int *m_match_array;
        float* m_dist2_array;
        float* m_sdot_array;
        float* m_cdot_array;
        const float3 *m_points;
        const float *m_shape_angles;
        const float *m_comp_angles;
        const unsigned int m_Np;
        const trajectory::Box m_box;
        const float m_rmax;
        const float m_shape_dot_target;
        const float m_shape_dot_tol;
        const float m_comp_dot_target;
        const float m_comp_dot_tol;

    public:
        ComputePairing2DWithoutCellList(unsigned int *match_array,
                                        float* dist2_array,
                                        float* sdot_array,
                                        float* cdot_array,
                                        const float3 *points,
                                        const float *shape_angles,
                                        const float *comp_angles,
                                        const unsigned int Np,
                                        const float r_max,
                                        const trajectory::Box box,
                                        const float shape_dot_target,
                                        const float shape_dot_tol,
                                        const float comp_dot_target,
                                        const float comp_dot_tol)
        : m_match_array(match_array), m_dist2_array(dist2_array), m_sdot_array(sdot_array), m_cdot_array(cdot_array),
          m_points(points), m_shape_angles(shape_angles), m_comp_angles(comp_angles), m_Np(Np), m_rmax(r_max),
          m_box(box), m_shape_dot_target(shape_dot_target), m_shape_dot_tol(shape_dot_tol),
          m_comp_dot_target(comp_dot_target), m_comp_dot_tol(comp_dot_tol)
        {
        }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // m_nP = Np;
            // for each particle
            float dist2;
            float sdot;
            float cdot;
            dist2 = 0.0;
            sdot = 0.0;
            cdot = 0.0;
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                float3 r_i = m_points[i];
                float angle_s_i = m_shape_angles[i];
                float angle_c_i = m_comp_angles[i];
                for (unsigned int j = 0; j < m_Np; j++)
                    {
                    float3 r_j = m_points[j];
                    float angle_s_j = m_shape_angles[j];
                    float angle_c_j = m_comp_angles[j];
                    // will skip same particle
                    if (i == j)
                        {
                        continue;
                        }

                    if (comp_check_2D(m_rmax,
                                   m_box,
                                   r_i,
                                   r_j,
                                   angle_s_i,
                                   angle_s_j,
                                   angle_c_i,
                                   angle_c_j,
                                   m_shape_dot_target,
                                   m_shape_dot_tol,
                                   m_comp_dot_target,
                                   m_comp_dot_tol,
                                   dist2,
                                   sdot,
                                   cdot))
                        {
                        m_match_array[i] = 1;
                        // m_match_array[j] = 1;
                        m_dist2_array[i] = dist2;
                        // m_dist2_array[j] = dist2;
                        m_sdot_array[i] = sdot;
                        // m_sdot_array[j] = sdot;
                        m_cdot_array[i] = cdot;
                        // m_cdot_array[j] = cdot;
                        }
                    } // done looping over check points
                } // done looping over reference points
            }
    };

bool pairing::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());
    if (m_box.is2D())
    l_min = fmin(l_min, m_box.getLy());
    if (m_rmax < l_min/3)
    return true;
    return false;
    }

void pairing::compute(unsigned int* match,
                      float* dist2,
                      float* sdots,
                      float* cdots,
                      const float3* points,
                      const float* shape_angles,
                      const float* comp_angles,
                      const unsigned int Np)
    {
    if (useCells())
        {
        m_lc->computeCellList(points, Np);
        parallel_for(blocked_range<size_t>(0,Np), ComputePairing2DCellList(match,
                                                                           dist2,
                                                                           sdots,
                                                                           cdots,
                                                                           points,
                                                                           shape_angles,
                                                                           comp_angles,
                                                                           Np,
                                                                           m_lc,
                                                                           m_rmax,
                                                                           m_box,
                                                                           m_shape_dot_target,
                                                                           m_shape_dot_tol,
                                                                           m_comp_dot_target,
                                                                           m_comp_dot_tol));
        }
    else
        {
        parallel_for(blocked_range<size_t>(0,Np), ComputePairing2DWithoutCellList(match,
                                                                                  dist2,
                                                                                  sdots,
                                                                                  cdots,
                                                                                  points,
                                                                                  shape_angles,
                                                                                  comp_angles,
                                                                                  Np,
                                                                                  m_rmax,
                                                                                  m_box,
                                                                                  m_shape_dot_target,
                                                                                  m_shape_dot_tol,
                                                                                  m_comp_dot_target,
                                                                                  m_comp_dot_tol));
        }
    }

void pairing::compute(unsigned int* match,
                      float* dist2,
                      float* sdots,
                      float* cdots,
                      const float3* points,
                      const float4* shape_quats,
                      const float4* comp_quats,
                      const unsigned int Np)
    {
    if (useCells())
        {
        m_lc->computeCellList(points, Np);
        parallel_for(blocked_range<size_t>(0,Np), ComputePairing3DCellList(match,
                                                                           dist2,
                                                                           sdots,
                                                                           cdots,
                                                                           points,
                                                                           shape_quats,
                                                                           comp_quats,
                                                                           Np,
                                                                           m_lc,
                                                                           m_rmax,
                                                                           m_box,
                                                                           m_shape_dot_target,
                                                                           m_shape_dot_tol,
                                                                           m_comp_dot_target,
                                                                           m_comp_dot_tol));
        }
    else
        {
        parallel_for(blocked_range<size_t>(0,Np), ComputePairing3DWithoutCellList(match,
                                                                                  dist2,
                                                                                  sdots,
                                                                                  cdots,
                                                                                  points,
                                                                                  shape_quats,
                                                                                  comp_quats,
                                                                                  Np,
                                                                                  m_rmax,
                                                                                  m_box,
                                                                                  m_shape_dot_target,
                                                                                  m_shape_dot_tol,
                                                                                  m_comp_dot_target,
                                                                                  m_comp_dot_tol));
        }
    }

void pairing::computePy(boost::python::numeric::array match,
                        boost::python::numeric::array dist2,
                        boost::python::numeric::array sdots,
                        boost::python::numeric::array cdots,
                        boost::python::numeric::array points,
                        boost::python::numeric::array shape_orientations,
                        boost::python::numeric::array comp_orientations)
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
    num_util::check_type(match, PyArray_INT);
    num_util::check_rank(match, 1);
    num_util::check_type(dist2, PyArray_FLOAT);
    num_util::check_rank(dist2, 1);
    num_util::check_type(sdots, PyArray_FLOAT);
    num_util::check_rank(sdots, 1);
    num_util::check_type(cdots, PyArray_FLOAT);
    num_util::check_rank(cdots, 1);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(shape_orientations, PyArray_FLOAT);
    unsigned int orientation_rank = num_util::rank(shape_orientations);
    num_util::check_type(comp_orientations, PyArray_FLOAT);
    // num_util::check_rank(comp_orientations, 1);

    // get the number of particles
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    const unsigned int Np = num_util::shape(points)[0];

    //validate that the types and angles coming in are the correct size
    num_util::check_dim(shape_orientations, 0, Np);
    num_util::check_dim(comp_orientations, 0, Np);

    // get the raw data pointers and compute the cell list
    unsigned int* match_raw = (unsigned int*) num_util::data(match);
    float* dist2_raw = (float*) num_util::data(dist2);
    float* sdots_raw = (float*) num_util::data(sdots);
    float* cdots_raw = (float*) num_util::data(cdots);
    const float3* points_raw = (float3*) num_util::data(points);
    if (orientation_rank == 2)
        {
        num_util::check_dim(shape_orientations, 2, 4);
        const float4* shape_orientations_raw = (float4*) num_util::data(shape_orientations);
        const float4* comp_orientations_raw = (float4*) num_util::data(comp_orientations);
        compute(match_raw,
                dist2_raw,
                sdots_raw,
                cdots_raw,
                points_raw,
                shape_orientations_raw,
                comp_orientations_raw,
                Np);
        }
    else
        {
        const float* shape_angles_raw = (float*) num_util::data(shape_orientations);
        const float* comp_angles_raw = (float*) num_util::data(comp_orientations);
        compute(match_raw,
                dist2_raw,
                sdots_raw,
                cdots_raw,
                points_raw,
                shape_angles_raw,
                comp_angles_raw,
                Np);
        }
    }

void export_pairing()
    {
    class_<pairing>("pairing", init<trajectory::Box&, float, float, float, float, float>())
        .def("getBox", &pairing::getBox, return_internal_reference<>())
        .def("compute", &pairing::computePy)
        ;
    }

}; }; // end namespace freud::pairing
