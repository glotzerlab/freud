#include "BondOrder.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace boost::python;
using namespace tbb;

/*! \file BondOrder.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

BondOrder::BondOrder(float rmax, float k, unsigned int n, unsigned int nbins_t, unsigned int nbins_p)
    : m_box(trajectory::Box()), m_rmax(rmax), m_k(k), m_nbins_t(nbins_t), m_nbins_p(nbins_p), m_Np(0)
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
    delete m_nn;
    }

class ComputeBondOrder
    {
    private:
        const trajectory::Box& m_box;
        const float m_rmax;
        const float m_k;
        const locality::NearestNeighbors *m_nn;
        const vec3<float> *m_points;
        std::complex<float> *m_psi_array;
    public:
        ComputeBondOrder(std::complex<float> *psi_array,
                                 const trajectory::Box& box,
                                 const float rmax,
                                 const float k,
                                 const locality::NearestNeighbors *nn,
                                 const vec3<float> *points)
            : m_box(box), m_rmax(rmax), m_k(k), m_nn(nn), m_points(points), m_psi_array(psi_array)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float rmaxsq = m_rmax * m_rmax;

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                m_psi_array[i] = 0;
                vec3<float> ref = m_points[i];

                //loop over neighbors
                locality::NearestNeighbors::iteratorneighbor it = m_nn->iterneighbor(i);
                for (unsigned int j = it.begin(); !it.atEnd(); j = it.next())
                    {

                    //compute r between the two particles
                    vec3<float> delta = m_box.wrap(m_points[j] - ref);

                    float rsq = dot(delta, delta);
                    if (rsq > 1e-6)
                        {
                        //compute psi for neighboring particle(only constructed for 2d)
                        float psi_ij = atan2f(delta.y, delta.x);
                        m_psi_array[i] += exp(complex<float>(0,m_k*psi_ij));
                        }
                    }
                m_psi_array[i] /= complex<float>(m_k);
                }
            }
    };

void BondOrder::compute(const vec3<float> *points, unsigned int Np)
    {
    // compute the cell list
    m_nn->compute(m_box,points,Np,points,Np);
    m_nn->setRMax(m_rmax);

    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_psi_array = boost::shared_array<complex<float> >(new complex<float> [Np]);
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Np), ComputeBondOrder(m_psi_array.get(), m_box, m_rmax, m_k, m_nn, points));

    // save the last computed number of particles
    m_Np = Np;
    }

void BondOrder::computePy(trajectory::Box& box,
                                  boost::python::numeric::array points)
    {
    //validate input type and rank
    m_box = box;
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute order parameter
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);

        // compute the order parameter with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np);
        }
    }

void export_BondOrder()
    {
    class_<BondOrder>("BondOrder", init<float>())
        .def(init<float, float>())
        .def(init<float, float, unsigned int>())
        .def("getBox", &BondOrder::getBox, return_internal_reference<>())
        .def("compute", &BondOrder::computePy)
        .def("getPsi", &BondOrder::getPsiPy)
        ;
    }

}; }; // end namespace freud::order


