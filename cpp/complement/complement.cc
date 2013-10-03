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

complement::complement(const trajectory::Box& box, float rmax, float dot_target, float dot_tol)
    : m_box(box), m_rmax(rmax), m_dot_target(dot_target), m_dot_tol(dot_tol)
    {
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (rmax > box.getLx()/2 || rmax > box.getLy()/2)
    throw invalid_argument("rmax must be smaller than half the smallest box size");

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

float complement::dot3(float3 v1, float3 v2)
    {
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    }

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

float2 complement::into_local(float2 ref_point,
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

float complement::cavity_depth(float2 t[])
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

void complement::compute(unsigned int* match,
                float3* points,
                float* angles,
                unsigned int Np)
    {
    m_nmatch = 0;
    if (useCells())
        {
        computeWithCellList(match,
                            points,
                            angles,
                            Np);
        }
    else
        {
        computeWithoutCellList(match,
                            points,
                            angles,
                            Np);
        }
    }

void complement::computeWithoutCellList(unsigned int* match,
                float3* points,
                float* angles,
                unsigned int Np)
    {
    m_nP = Np;
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
            float p_angle = angles[i];
            for (unsigned int j = 0; j < m_nP; j++)
                {
                float3 check = points[j];
                float c_angle = angles[j];

                float2 r_ij;
                r_ij.x = point.x - check.x;
                r_ij.y = point.y - check.y;
                float2 theta_i;
                float2 theta_j;
                theta_i.x = cosf(p_angle);
                theta_i.y = sinf(p_angle);
                theta_j.x = cosf(c_angle);
                theta_j.y = sinf(c_angle);

                float d_ij = dot2(r_ij, r_ij);
                float theta_ij = dot2(theta_i, theta_j);

                if ((d_ij < rmaxsq) && (theta_ij > (m_dot_target - m_dot_tol)) && (theta_ij < (m_dot_target + m_dot_tol)))
                    {
                    match[i] = 1;
                    match[j] = 1;
                    m_nmatch++;
                    }
                } // done looping over check
            } //done looping over ref
        } // End of parallel section
    }

void complement::computeWithCellList(unsigned int* match,
                float3* points,
                float* angles,
                unsigned int Np)
    {
    m_nP = Np;
    m_lc->computeCellList(points, m_nP);
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
            float p_angle = angles[i];
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
                    float c_angle = angles[j];
                    // will skip same particle
                    // is this necessary?
                    // if (i == j)
                    //     {
                    //     continue;
                    //     }

                    // new code here
                    float2 theta_i;
                    float2 theta_j;
                    theta_i.x = cosf(p_angle);
                    theta_i.y = sinf(p_angle);
                    theta_j.x = cosf(c_angle);
                    theta_j.y = sinf(c_angle);

                    float theta_ij = dot2(theta_i, theta_j);

                    if ((theta_ij > (m_dot_target - m_dot_tol)) && (theta_ij < (m_dot_target + m_dot_tol)))
                        {
                        match[i] = 1;
                        match[j] = 1;
                        m_nmatch++;
                        }
                    } // done looping over neighbors
                } // done looping over neighbor cells
            } // done looping over reference points
        } // End of parallel section
    }

void complement::computePy(boost::python::numeric::array match,
                    boost::python::numeric::array points,
                    boost::python::numeric::array angles)
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
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(angles, PyArray_FLOAT);
    num_util::check_rank(angles, 1);

    // get the number of particles
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    //validate that the types and angles coming in are the correct size
    num_util::check_dim(angles, 0, Np);

    // get the raw data pointers and compute the cell list
    unsigned int* match_raw = (unsigned int*) num_util::data(match);
    float3* points_raw = (float3*) num_util::data(points);
    float* angles_raw = (float*) num_util::data(angles);

    compute(match_raw,
            points_raw,
            angles_raw,
            Np);
    }

void export_complement()
    {
    class_<complement>("complement", init<trajectory::Box&, float, float, float>())
        .def("getBox", &complement::getBox, return_internal_reference<>())
        .def("compute", &complement::computePy)
        .def("getNpair", &complement::getNpairPy)
        .def("_sameSide", &complement::_sameSidePy)
        .def("_isInside", &complement::_isInsidePy)
        .def("_cross", &complement::_crossPy)
        .def("_dot3", &complement::_dotPy)
        .def("_mat_rot", &complement::_mat_rotPy)
        .def("_into_local", &complement::_into_localPy)
        ;
    }

}; }; // end namespace freud::complement
