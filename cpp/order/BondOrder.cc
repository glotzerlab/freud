#include "BondOrder.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdexcept>
#include <complex>

using namespace std;
using namespace tbb;

/*! \file BondOrder.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

BondOrder::BondOrder(float rmax, float k, unsigned int n, unsigned int nbins_t, unsigned int nbins_p)
    : m_box(trajectory::Box()), m_rmax(rmax), m_k(k), m_nbins_t(nbins_t), m_nbins_p(nbins_p), m_Np(0),
      m_frame_counter(0)
    {
    // sanity checks, but this is actually kinda dumb if these values are 1
    if (nbins_t < 1)
        throw invalid_argument("must be at least 1 bin in theta");
    if (nbins_p < 1)
        throw invalid_argument("must be at least 1 bin in p");
    // calculate dt, dp
    /*
    0 < \theta < 2PI; 0 < \phi < PI
    */
    m_dt = 2.0 * M_PI / float(m_nbins_t);
    m_dp = M_PI / float(m_nbins_p);
    // this shouldn't be able to happen, but it's always better to check
    if (m_dt > 2.0 * M_PI)
        throw invalid_argument("2PI must be greater than dt");
    if (m_dp > M_PI)
        throw invalid_argument("PI must be greater than dp");

    // precompute the bin center positions for t
    m_theta_array = boost::shared_array<float>(new float[m_nbins_t]);
    for (unsigned int i = 0; i < m_nbins_t; i++)
        {
        float t = float(i) * m_dt;
        float nextt = float(i+1) * m_dt;
        m_theta_array[i] = ((t + nextt) / 2.0);
        }

    // precompute the bin center positions for p
    m_phi_array = boost::shared_array<float>(new float[m_nbins_p]);
    for (unsigned int i = 0; i < m_nbins_p; i++)
        {
        float p = float(i) * m_dp;
        float nextp = float(i+1) * m_dp;
        m_phi_array[i] = ((p + nextp) / 2.0);
        }

    // precompute the surface area array
    m_sa_array = boost::shared_array<float>(new float[m_nbins_t*m_nbins_p]);
    memset((void*)m_sa_array.get(), 0, sizeof(float)*m_nbins_t*m_nbins_p);
    Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);
    for (unsigned int i = 0; i < m_nbins_t; i++)
        {
        float theta = (float)i * m_dt;
        for (unsigned int j = 0; j < m_nbins_p; j++)
            {
            float phi = (float)j * m_dp;
            float sa = m_dt * (cos(phi) - cos(phi + m_dp));
            m_sa_array[sa_i((int)i, (int)j)] = sa;
            }
        }

    // initialize the bin counts
    m_bin_counts = boost::shared_array<unsigned int>(new unsigned int[m_nbins_t*m_nbins_p]);
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);

    // initialize the bond order array
    m_bo_array = boost::shared_array<float>(new float[m_nbins_t*m_nbins_p]);
    memset((void*)m_bin_counts.get(), 0, sizeof(float)*m_nbins_t*m_nbins_p);

    // create NearestNeighbors object
    // if n is zero, set the number of neighbors to k
    // otherwise set to n
    // this is super dangerous...
    m_nn = new locality::NearestNeighbors(m_rmax, n==0? (unsigned int) k: n);
    }

BondOrder::~BondOrder()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_nn;
    }

class CombineBondOrder
    {
    private:
        unsigned int m_nbins_t;
        unsigned int m_nbins_p;
        unsigned int *m_bin_counts;
        float *m_bo_array;
        float *m_sa_array;
        tbb::enumerable_thread_specific<unsigned int *>& m_local_bin_counts;
    public:
        CombineBondOrder(unsigned int nbins_t,
                         unsigned int nbins_p,
                         unsigned int *bin_counts,
                         float *bo_array,
                         float *sa_array,
                         tbb::enumerable_thread_specific<unsigned int *>& local_bin_counts)
            : m_nbins_t(nbins_t), m_nbins_p(nbins_p), m_bin_counts(bin_counts), m_bo_array(bo_array), m_sa_array(sa_array),
              m_local_bin_counts(local_bin_counts)
        {
        }
        void operator()( const blocked_range<size_t> &myBin ) const
            {
            Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);
            for (size_t i = myBin.begin(); i != myBin.end(); i++)
                {
                for (size_t j = 0; j < m_nbins_p; j++)
                    {
                    for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                         local_bins != m_local_bin_counts.end(); ++local_bins)
                        {
                        m_bin_counts[sa_i((int)i, (int)j)] += (*local_bins)[sa_i((int)i, (int)j)];
                        }
                    m_bo_array[sa_i((int)i, (int)j)] = m_bin_counts[sa_i((int)i, (int)j)] / m_sa_array[sa_i((int)i, (int)j)];
                    }
                }
            }
    };

class ComputeBondOrder
    {
    private:
        tbb::enumerable_thread_specific<unsigned int *>& m_local_bin_counts;
        unsigned int m_nbins_t;
        unsigned int m_nbins_p;
        const trajectory::Box& m_box;
        const float m_rmax;
        const float m_k;
        const float m_dt;
        const float m_dp;
        const locality::NearestNeighbors *m_nn;
        const vec3<float> *m_ref_points;
        const quat<float> *m_ref_orientations;
        const unsigned int m_Nref;
        const vec3<float> *m_points;
        const quat<float> *m_orientations;
        const unsigned int m_Np;
    public:
        ComputeBondOrder(tbb::enumerable_thread_specific<unsigned int *>& local_bin_counts,
                         unsigned int nbins_t,
                         unsigned int nbins_p,
                         const trajectory::Box& box,
                         const float rmax,
                         const float k,
                         const float dt,
                         const float dp,
                         const locality::NearestNeighbors *nn,
                         const vec3<float> *ref_points,
                         const quat<float> *ref_orientations,
                         const unsigned int Nref,
                         const vec3<float> *points,
                         const quat<float> *orientations,
                         const unsigned int Np)
            : m_local_bin_counts(local_bin_counts), m_nbins_t(nbins_t), m_nbins_p(nbins_p), m_box(box), m_rmax(rmax),
              m_k(k), m_dt(dt), m_dp(dp), m_nn(nn), m_ref_points(ref_points), m_ref_orientations(ref_orientations), m_Nref(Nref),
              m_points(points), m_orientations(orientations), m_Np(Np)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float dt_inv = 1.0f / m_dt;
            float dp_inv = 1.0f / m_dp;
            float rmaxsq = m_rmax * m_rmax;
            Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_nbins_t*m_nbins_p];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);
                }

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                vec3<float> ref_pos = m_ref_points[i];
                quat<float> ref_q(m_ref_orientations[i]);

                //loop over neighbors
                locality::NearestNeighbors::iteratorneighbor it = m_nn->iterneighbor(i);
                for (unsigned int j = it.begin(); !it.atEnd(); j = it.next())
                    {

                    //compute r between the two particles
                    vec3<float> delta = m_box.wrap(m_points[j] - ref_pos);

                    float rsq = dot(delta, delta);
                    if (rsq > 1e-6)
                        {
                        //compute psi for neighboring particle(only constructed for 2d)
                        // get orientation
                        // I don't think this is needed
                        // quat<float> orient(m_orientations[j]);
                        vec3<float> v(delta);
                        v = rotate(conj(ref_q), v);
                        // get theta, phi
                        float theta = atan2f(v.y, v.x);
                        theta = (theta < 0) ? theta+2*M_PI : theta;
                        theta = (theta > 2*M_PI) ? theta-2*M_PI : theta;
                        float phi = atan2f(sqrt(v.x*v.x + v.y*v.y), v.z);
                        phi = (phi < 0) ? phi+2*M_PI : phi;
                        phi = (phi > 2*M_PI) ? phi-2*M_PI : phi;
                        // bin the point
                        float bint = floorf(theta * dt_inv);
                        float binp = floorf(phi * dp_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibint = _mm_cvtt_ss2si(_mm_load_ss(&bint));
                        unsigned int ibinp = _mm_cvtt_ss2si(_mm_load_ss(&binp));
                        #else
                        unsigned int ibint = (unsigned int)(bint);
                        unsigned int ibinp = (unsigned int)(binp);
                        #endif

                        // increment the bin
                        if ((ibint < m_nbins_t) && (ibinp < m_nbins_p))
                            {
                            ++m_local_bin_counts.local()[sa_i(ibint, ibinp)];
                            }
                        }
                    }
                }
            }
    };

void BondOrder::reduceBondOrder()
    {
    memset((void*)m_bo_array.get(), 0, sizeof(float)*m_nbins_t*m_nbins_p);
    parallel_for(blocked_range<size_t>(0,m_nbins_t),
                 CombineBondOrder(m_nbins_t,
                                  m_nbins_p,
                                  m_bin_counts.get(),
                                  m_bo_array.get(),
                                  m_sa_array.get(),
                                  m_local_bin_counts));
    Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);
    for (unsigned int i=0; i<m_nbins_t; i++)
        {
        for (unsigned int j=0; j<m_nbins_p; j++)
            {
            m_bin_counts[sa_i((int)i, (int)j)] = m_bin_counts[sa_i((int)i, (int)j)] / (float)m_frame_counter;
            m_bo_array[sa_i((int)i, (int)j)] = m_bo_array[sa_i((int)i, (int)j)] / (float)m_frame_counter;
            }
        }
    }

boost::shared_array<float> BondOrder::getBondOrder()
    {
    reduceBondOrder();
    return m_bo_array;
    }

// boost::python::numeric::array BondOrder::getBondOrderPy()
//     {
//     reduceBondOrder();
//     float *arr = m_bo_array.get();
//     std::vector<intp> dims(2);
//     dims[0] = m_nbins_p;
//     dims[1] = m_nbins_t;
//     return num_util::makeNum(arr, dims);
//     }

void BondOrder::resetBondOrder()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);
        }
    // reset the frame counter
    m_frame_counter = 0;
    }

void BondOrder::accumulate(vec3<float> *ref_points,
                           quat<float> *ref_orientations,
                           unsigned int Nref,
                           vec3<float> *points,
                           quat<float> *orientations,
                           unsigned int Np)
    {
    // compute the cell list
    m_nn->compute(m_box,ref_points,Nref,points,Np);
    m_nn->setRMax(m_rmax);

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Nref),
                 ComputeBondOrder(m_local_bin_counts,
                                  m_nbins_t,
                                  m_nbins_p,
                                  m_box,
                                  m_rmax,
                                  m_k,
                                  m_dt,
                                  m_dp,
                                  m_nn,
                                  ref_points,
                                  ref_orientations,
                                  Nref,
                                  points,
                                  orientations,
                                  Np));

    // save the last computed number of particles
    m_Nref = Nref;
    m_Np = Np;
    m_frame_counter++;
    }

// void BondOrder::accumulatePy(trajectory::Box& box,
//                              boost::python::numeric::array ref_points,
//                              boost::python::numeric::array ref_orientations,
//                              boost::python::numeric::array points,
//                              boost::python::numeric::array orientations)
//     {
//     //validate input type and rank
//     m_box = box;
//     num_util::check_type(ref_points, NPY_FLOAT);
//     num_util::check_rank(ref_points, 2);
//     num_util::check_type(ref_orientations, NPY_FLOAT);
//     num_util::check_rank(ref_orientations, 2);
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);
//     num_util::check_type(orientations, NPY_FLOAT);
//     num_util::check_rank(orientations, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     num_util::check_dim(ref_points, 1, 3);
//     unsigned int Nref = num_util::shape(ref_points)[0];

//     // check the size of angles to be correct
//     num_util::check_dim(ref_orientations, 0, Nref);
//     num_util::check_dim(ref_orientations, 1, 4);
//     num_util::check_dim(orientations, 0, Np);
//     num_util::check_dim(orientations, 1, 4);

//     // get the raw data pointers and compute order parameter
//     vec3<float>* ref_points_raw = (vec3<float>*) num_util::data(ref_points);
//     quat<float>* ref_orientations_raw = (quat<float>*) num_util::data(ref_orientations);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     quat<float>* orientations_raw = (quat<float>*) num_util::data(orientations);

//         // compute the order parameter with the GIL released
//         {
//         util::ScopedGILRelease gil;
//         accumulate(ref_points_raw,
//                    ref_orientations_raw,
//                    Nref,
//                    points_raw,
//                    orientations_raw,
//                    Np);
//         }
//     }

// //! \internal
// /*! \brief Exposed function to python to calculate the PMF
// */
// void BondOrder::computePy(trajectory::Box& box,
//                           boost::python::numeric::array ref_points,
//                           boost::python::numeric::array ref_orientations,
//                           boost::python::numeric::array points,
//                           boost::python::numeric::array orientations)
//     {
//     // validate input type and rank
//     resetBondOrder();
//     accumulatePy(box, ref_points, ref_orientations, points, orientations);
//     }

// void export_BondOrder()
//     {
//     class_<BondOrder>("BondOrder", init<float, float, unsigned int, unsigned int, unsigned int>())
//         .def("getBox", &BondOrder::getBox, return_internal_reference<>())
//         .def("accumulate", &BondOrder::accumulatePy)
//         .def("compute", &BondOrder::computePy)
//         .def("getBondOrder", &BondOrder::getBondOrderPy)
//         .def("getTheta", &BondOrder::getThetaPy)
//         .def("getPhi", &BondOrder::getPhiPy)
//         .def("resetBondOrder", &BondOrder::resetBondOrderPy)
//         ;
//     }

}; }; // end namespace freud::order
