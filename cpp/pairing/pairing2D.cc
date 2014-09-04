#include "pairing2D.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <complex>
#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;

using namespace tbb;

namespace freud { namespace pairing {

pairing::pairing(const trajectory::Box& box,
                 const float rmax,
                 const unsigned int k,
                 const float comp_dot_tol)
    : m_box(box), m_rmax(rmax), m_k(k), m_nn(box, rmax, (unsigned int) k), m_Np(0), m_No(0), m_comp_dot_tol(comp_dot_tol)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_match_array = boost::shared_array<unsigned int>(new unsigned int[m_Np]);
    for (unsigned int i = 0; i < m_Np; i++)
        {
        m_match_array[i] = 0;
        }
    // create the pairing array to store particle pairs
    // m_match_array[i] will have the pair of particle i stored at idx=i
    // if there is no pairing, it will store itself
    m_pair_array = boost::shared_array<unsigned int>(new unsigned int[m_Np]);
    for (unsigned int i = 0; i < m_Np; i++)
        {
        m_pair_array[i] = i;
        }
    }

// class ComputePairing2D
//     {
//     private:
//         atomic<unsigned int> *m_match_array;
//         atomic<unsigned int> *m_pair_array;
//         const float3 *m_points;
//         const float *m_orientations;
//         const float *m_comp_orientations;
//         const unsigned int m_Np;
//         const unsigned int m_No;
//         const locality::NearestNeighbors &m_nn;
//         const unsigned int m_k;
//         const float m_rmax;
//         const trajectory::Box m_box;
//         const float m_comp_dot_tol;

//     public:
//         ComputePairing2D(atomic<unsigned int> *match_array,
//                          atomic<unsigned int> *pair_array,
//                          const float3 *points,
//                          const float *orientations,
//                          const float *comp_orientations,
//                          const unsigned int Np,
//                          const unsigned int No,
//                          const locality::NearestNeighbors &nn,
//                          const unsigned int k,
//                          const float rmax,
//                          const trajectory::Box box,
//                          const float comp_dot_tol)
//         : m_match_array(match_array), m_pair_array(pair_array), m_points(points), m_orientations(orientations),
//           m_comp_orientations(comp_orientations), m_Np(Np), m_No(No), m_nn(nn), m_k(k), m_rmax(rmax), m_box(box),
//           m_comp_dot_tol(comp_dot_tol)
//         {
//         }

//         void operator()( const blocked_range<size_t>& r ) const
//             {
//             // for each particle
//             for (size_t i = r.begin(); i != r.end(); i++)
//                 {
//                 const vec3<float> r_i(m_points[i].x, m_points[i].y, m_points[i].z);
//                 // get the neighbors of i
//                 boost::shared_array<unsigned int> neighbors = m_nn.getNeighbors(i);
//                 // loop over all neighboring particles
//                 for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
//                     {
//                     unsigned int j = neighbors[neigh_idx];
//                     const vec3<float> r_j(m_points[j].x, m_points[j].y, m_points[j].z);
//                     vec3<float> r_ij(r_j - r_i);
//                     vec3<float> r_ji(r_i - r_j);
//                     float3 wrapped(m_box.wrap(make_float3(r_ij.x, r_ij.y, r_ij.z)));
//                     r_ij = vec3<float>(wrapped.x, wrapped.y, wrapped.z);
//                     wrapped = m_box.wrap(make_float3(r_ji.x, r_ji.y, r_ji.z));
//                     r_ji = vec3<float>(wrapped.x, wrapped.y, wrapped.z);
//                     const float rsq(dot(r_ij, r_ij));

//                     // will skip same particle
//                     if (rsq > 1e-6)
//                         {
//                         // check if the particles are paired
//                         // particles are paired if they are the nearest neighbors that have the complementary vector
//                         // pointing in the same direction as the interparticle vector

//                         // for each potential complementary orientation for particle i
//                         for (unsigned int a=0; a<m_No; a++)
//                             {
//                             // generate vectors
//                             std::complex<float> tmp_i = std::polar<float>(1.0, m_orientations[i] + m_comp_orientations[i*m_No + a]);
//                             vec3<float> c_i;
//                             c_i.x = std::real<float>(tmp_i);
//                             c_i.y = std::imag<float>(tmp_i);
//                             c_i.z = 0;

//                             // for each potential complementary orientation for particle j
//                             for (unsigned int b=0; b<m_No; b++)
//                                 {
//                                 std::complex<float> tmp_j = std::polar<float>(1.0, m_orientations[j] + m_comp_orientations[j*m_No + b]);
//                                 vec3<float> c_j;
//                                 c_j.x = std::real<float>(tmp_j);
//                                 c_j.y = std::imag<float>(tmp_j);
//                                 c_j.z = 0;
//                                 // calculate the dot products
//                                 float d_ij = dot(c_i, r_ij);
//                                 float d_ji = dot(c_j, r_ji);
//                                 if ((abs(d_ij - 1.0) < m_comp_dot_tol) && (abs(d_ji - 1.0) < m_comp_dot_tol))
//                                     {
//                                     m_match_array[i] = 1;
//                                     m_match_array[j] = 1;
//                                     m_pair_array[i] = j;
//                                     m_pair_array[j] = i;
//                                     continue;
//                                     }
//                                 }
//                             }

//                         // if (comp_check_2D(m_rmax,
//                         //                m_box,
//                         //                r_i,
//                         //                r_j,
//                         //                angle_s_i,
//                         //                angle_s_j,
//                         //                angle_c_i,
//                         //                angle_c_j,
//                         //                m_shape_dot_target,
//                         //                m_shape_dot_tol,
//                         //                m_comp_dot_target,
//                         //                m_comp_dot_tol,
//                         //                dist2,
//                         //                sdot,
//                         //                cdot))
//                         //     {
//                         //     m_match_array[i] = 1;
//                         //     // m_match_array[j] = 1;
//                         //     m_dist2_array[i] = dist2;
//                         //     // m_dist2_array[j] = dist2;
//                         //     m_sdot_array[i] = sdot;
//                         //     // m_sdot_array[j] = sdot;
//                         //     m_cdot_array[i] = cdot;
//                         //     // m_cdot_array[j] = cdot;
//                         //     }

//                         }
//                     } // done looping over neighbors
//                 } // done looping over reference points
//             }
//     };

void pairing::ComputePairing2D(const float3 *points,
                               const float *orientations,
                               const float *comp_orientations,
                               const unsigned int Np,
                               const unsigned int No)
    {
    // for each particle
    for (size_t i = 0; i < m_Np; i++)
        {
        if (m_match_array[i] == 1)
            continue;
        const vec3<float> r_i(points[i].x, points[i].y, points[i].z);
        // get the neighbors of i
        boost::shared_array<unsigned int> neighbors = m_nn.getNeighbors(i);
        // loop over all neighboring particles
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            bool is_finished = false;
            unsigned int j = neighbors[neigh_idx];
            const vec3<float> r_j(points[j].x, points[j].y, points[j].z);
            vec3<float> r_ij(r_j - r_i);
            vec3<float> r_ji(r_i - r_j);
            float3 wrapped(m_box.wrap(make_float3(r_ij.x, r_ij.y, r_ij.z)));
            r_ij = vec3<float>(wrapped.x, wrapped.y, wrapped.z);
            wrapped = m_box.wrap(make_float3(r_ji.x, r_ji.y, r_ji.z));
            r_ji = vec3<float>(wrapped.x, wrapped.y, wrapped.z);
            const float rsq(dot(r_ij, r_ij));

            // will skip same particle
            if (rsq > 1e-6)
                {
                // check if the particles are paired
                // particles are paired if they are the nearest neighbors that have the complementary vector
                // pointing in the same direction as the interparticle vector

                // for each potential complementary orientation for particle i
                for (unsigned int a=0; a<m_No; a++)
                    {
                    // generate vectors
                    std::complex<float> tmp_i = std::polar<float>(1.0, orientations[i] + comp_orientations[i*m_No + a]);
                    vec3<float> c_i;
                    c_i.x = std::real<float>(tmp_i);
                    c_i.y = std::imag<float>(tmp_i);
                    c_i.z = 0;

                    // for each potential complementary orientation for particle j
                    for (unsigned int b=0; b<m_No; b++)
                        {
                        std::complex<float> tmp_j = std::polar<float>(1.0, orientations[j] + comp_orientations[j*m_No + b]);
                        vec3<float> c_j;
                        c_j.x = std::real<float>(tmp_j);
                        c_j.y = std::imag<float>(tmp_j);
                        c_j.z = 0;
                        // calculate the dot products
                        float d_ij = dot(c_i, r_ij);
                        float d_ji = dot(c_j, r_ji);
                        if ((abs(d_ij - 1.0) < m_comp_dot_tol) && (abs(d_ji - 1.0) < m_comp_dot_tol))
                            {
                            m_match_array[i] = 1;
                            m_match_array[j] = 1;
                            m_pair_array[i] = j;
                            m_pair_array[j] = i;
                            is_finished = true;
                            }
                        if (is_finished == true)
                            break;
                        }
                    if (is_finished == true)
                        break;
                    }
                if (is_finished == true)
                    break;
                }
            if (is_finished == true)
                break;
            } // done looping over neighbors
        } // done looping over reference points
    }

void pairing::compute(const float3* points,
                      const float* orientations,
                      const float* comp_orientations,
                      const unsigned int Np,
                      const unsigned int No)
    {
    m_nn.compute((vec3<float>*)points,Np);
    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_match_array = boost::shared_array<unsigned int>(new unsigned int[Np]);
        m_pair_array = boost::shared_array<unsigned int>(new unsigned int[Np]);
        }
    // reset the arrays
    for (unsigned int i = 0; i < Np; i++)
        {
        m_match_array[i] = 0;
        }
    for (unsigned int i = 0; i < Np; i++)
        {
        m_pair_array[i] = i;
        }
    // parallel_for(blocked_range<size_t>(0,Np),
    //              ComputePairing2D((atomic<unsigned int>*)m_match_array.get(),
    //                               (atomic<unsigned int>*)m_pair_array.get(),
    //                               points,
    //                               orientations,
    //                               comp_orientations,
    //                               Np,
    //                               No,
    //                               m_nn,
    //                               m_k,
    //                               m_rmax,
    //                               m_box,
    //                               m_comp_dot_tol));
     ComputePairing2D(points,
                      orientations,
                      comp_orientations,
                      Np,
                      No);
    m_Np = Np;
    m_No = No;
    }

void pairing::computePy(boost::python::numeric::array points,
                        boost::python::numeric::array orientations,
                        boost::python::numeric::array comp_orientations)
    {
    // points contains all the particle positions; Np x 3
    // types contains all the types; Np (x 1)
    // orientations contains the angle of each particle; Np (x1)
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(orientations, PyArray_FLOAT);
    num_util::check_rank(orientations, 1);
    num_util::check_type(comp_orientations, PyArray_FLOAT);
    num_util::check_rank(comp_orientations, 2);

    // get the number of particles
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    const unsigned int Np = num_util::shape(points)[0];

    //validate that the types and angles coming in are the correct size
    num_util::check_dim(orientations, 0, Np);
    num_util::check_dim(comp_orientations, 0, Np);
    const unsigned int No = num_util::shape(comp_orientations)[1];

    const float3* points_raw = (float3*) num_util::data(points);
    const float* orientations_raw = (float*) num_util::data(orientations);
    const float* comp_orientations_raw = (float*) num_util::data(comp_orientations);
    compute(points_raw,
            orientations_raw,
            comp_orientations_raw,
            Np,
            No);
    }

void export_pairing()
    {
    class_<pairing>("pairing", init<trajectory::Box&, float, unsigned int, float>())
        .def("getBox", &pairing::getBox, return_internal_reference<>())
        .def("compute", &pairing::computePy)
        .def("getMatch", &pairing::getMatchPy)
        .def("getPair", &pairing::getPairPy)
        ;
    }

}; }; // end namespace freud::pairing
