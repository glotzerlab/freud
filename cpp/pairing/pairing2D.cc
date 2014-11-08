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

pairing::pairing(const float rmax,
                 const unsigned int k,
                 const float comp_dot_tol)
    : m_box(trajectory::Box()), m_rmax(rmax), m_k(k), m_Np(0), m_No(0), m_comp_dot_tol(comp_dot_tol)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_match_array = boost::shared_array<unsigned int>(new unsigned int[m_Np]);
    for (unsigned int i = 0; i < m_Np; i++)
        {
        m_match_array[i] = 0;
        }
    // create the pairing array to store particle pairs
    // m_pair_array[i] will have the pair of particle i stored at idx=i
    // if there is no pairing, it will store itself
    m_pair_array = boost::shared_array<unsigned int>(new unsigned int[m_Np]);
    for (unsigned int i = 0; i < m_Np; i++)
        {
        m_pair_array[i] = i;
        }
    m_nn = new locality::NearestNeighbors();
    }

pairing::~pairing()
    {
    delete m_nn;
    }

void pairing::updateBox(trajectory::Box& box)
    {
    // check to make sure the provided box is valid
    if (m_rmax > box.getLx()/2 || m_rmax > box.getLy()/2)
        throw invalid_argument("rmax must be smaller than half the smallest box size");
    if (m_rmax > box.getLz()/2 && !box.is2D())
        throw invalid_argument("rmax must be smaller than half the smallest box size");
    // see if it is different than the current box
    if (m_box != box)
        {
        m_box = box;
        m_nn->updateBox(m_box, m_rmax, m_k);
        }
    }

void pairing::ComputePairing2D(const vec3<float> *points,
                               const float *orientations,
                               const float *comp_orientations,
                               const unsigned int Np,
                               const unsigned int No)
    {
    // for each particle
    Index2D b_i = Index2D(m_No, m_Np);
    for (unsigned int i = 0; i < m_Np; i++)
        {
        if (m_pair_array[i] != i)
            {
            continue;
            }
        const vec2<float> r_i(points[i].x, points[i].y);
        // get the neighbors of i
        boost::shared_array<unsigned int> neighbors = m_nn->getNeighbors(i);
        // loop over all neighboring particles
        bool is_paired = false;
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            unsigned int j = neighbors[neigh_idx];
            // need to check to make sure that neither i nor j are paired, as i could become paired in the inner loop
            if ((m_pair_array[j] != j) || (m_pair_array[i] != i))
                {
                break;
                }
            const vec2<float> r_j(points[j].x, points[j].y);
            vec2<float> r_ij(r_j - r_i);
            vec2<float> r_ji(r_i - r_j);
            vec3<float> delta = vec3<float>(r_ij.x, r_ij.y, 0.0);
            // float3 wrapped(m_box.wrap(make_float3(r_ij.x, r_ij.y, 0.0)));
            delta = m_box.wrap(delta);
            r_ij = vec2<float>(delta.x, delta.y);
            delta = vec3<float>(r_ji.x, r_ji.y, 0.0);
            delta = m_box.wrap(delta);
            r_ji = vec2<float>(delta.x, delta.y);
            float rsq(dot(r_ij, r_ij));

            // will skip same particle
            // shouldn't actually be needed
            if (rsq > 1e-6)
                {
                // check if the particles are paired
                // particles are paired if they are the nearest neighbors that have the complementary vector
                // pointing in the same direction as the interparticle vector

                // rotate the unit interparticle vector
                rotmat2<float> my_mat = rotmat2<float>::fromAngle(-orientations[i]);
                vec2<float> u_ij(r_ij/sqrt(rsq));
                u_ij = my_mat * u_ij;
                my_mat = rotmat2<float>::fromAngle(-orientations[j]);
                rsq = dot(r_ji, r_ji);
                vec2<float> u_ji(r_ji/sqrt(rsq));
                u_ji = my_mat * u_ji;

                // for each potential complementary orientation for particle i
                for (unsigned int a=0; a<m_No; a++)
                    {
                    if (is_paired == true)
                        {
                        break;
                        }
                    // generate vectors
                    float theta_ci = comp_orientations[b_i(a, i)];
                    vec2<float> c_i(cosf(theta_ci), sinf(theta_ci));

                    // for each potential complementary orientation for particle j
                    for (unsigned int b=0; b<m_No; b++)
                        {
                        if (is_paired == true)
                            break;
                        float theta_cj = comp_orientations[b_i(b, j)];
                        vec2<float> c_j(cosf(theta_cj), sinf(theta_cj));
                        // calculate the dot products
                        float d_ij = acos(dot(c_i, u_ij));
                        float d_ji = acos(dot(c_j, u_ji));
                        // this check assumes that the target angle between the interparticle vector and the complementary
                        // interface is 0. As the nearest neighbor list may use a larger rmax than was initialized, it has
                        // to check again
                        if ((d_ij < m_comp_dot_tol) && (d_ji < m_comp_dot_tol) && (is_paired==false) && (rsq < (m_rmax * m_rmax)))
                            {
                            m_match_array[i] = 1;
                            m_match_array[j] = 1;
                            m_pair_array[i] = j;
                            m_pair_array[j] = i;
                            is_paired = true;
                            } // done pairing particle
                        } // done checking all orientations of j
                    } // done checking all orientations of i
                } // done with not doing if the same particle (which should not happen)
            } // done looping over neighbors
        } // done looping over reference points
    }

void pairing::compute(const vec3<float>* points,
                      const float* orientations,
                      const float* comp_orientations,
                      const unsigned int Np,
                      const unsigned int No)
    {
    m_nn->compute(points,Np);
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
     ComputePairing2D(points,
                      orientations,
                      comp_orientations,
                      Np,
                      No);
    m_Np = Np;
    m_No = No;
    }

void pairing::computePy(trajectory::Box& box,
                        boost::python::numeric::array points,
                        boost::python::numeric::array orientations,
                        boost::python::numeric::array comp_orientations)
    {
    // points contains all the particle positions; Np x 3
    // orientations contains the orientations of each particle; Np (x1)
    // orientations contains the local orientations of possible interfaces; Np x No
    updateBox(box);
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

    // const float3* points_raw = (float3*) num_util::data(points);
    const vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
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
    class_<pairing>("pairing", init<float, unsigned int, float>())
        .def("getBox", &pairing::getBox, return_internal_reference<>())
        .def("compute", &pairing::computePy)
        .def("getMatch", &pairing::getMatchPy)
        .def("getPair", &pairing::getPairPy)
        ;
    }

}; }; // end namespace freud::pairing
