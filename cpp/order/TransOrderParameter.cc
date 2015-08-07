#include "TransOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace boost::python;
using namespace tbb;

/*! \file TransOrderParameter.h
    \brief Compute the translational order parameter for each particle
*/

namespace freud { namespace order {

TransOrderParameter::TransOrderParameter(float rmax, float k, unsigned int n)
    : m_box(trajectory::Box()), m_rmax(rmax), m_k(k), m_Np(0)
    {
    m_nn = new locality::NearestNeighbors(m_rmax, n==0? (unsigned int) k: n);
    }

TransOrderParameter::~TransOrderParameter()
    {
    delete m_nn;
    }

class ComputeTransOrderParameter
    {
    private:
        const trajectory::Box& m_box;
        const float m_rmax;
        const float m_k;
        const locality::NearestNeighbors *m_nn;
        const vec3<float> *m_points;
        std::complex<float> *m_dr_array;
    public:
        ComputeTransOrderParameter(std::complex<float> *dr_array,
                                 const trajectory::Box& box,
                                 const float rmax,
                                 const float k,
                                 const locality::NearestNeighbors *nn,
                                 const vec3<float> *points)
            : m_box(box), m_rmax(rmax), m_k(k), m_nn(nn), m_points(points), m_dr_array(dr_array)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float rmaxsq = m_rmax * m_rmax;

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                m_dr_array[i] = 0;
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
                        //compute dr for neighboring particle(only constructed for 2d)
                        m_dr_array[i] += complex<float>(delta.x, delta.y);
                        }
                    }
                m_dr_array[i] /= complex<float>(m_k);
                }
            }
    };

void TransOrderParameter::compute(const vec3<float> *points, unsigned int Np)
    {
    // compute the cell list
    m_nn->compute(m_box,points,Np,points,Np);
    m_nn->setRMax(m_rmax);

    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_dr_array = boost::shared_array<complex<float> >(new complex<float> [Np]);
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Np), ComputeTransOrderParameter(m_dr_array.get(), m_box, m_rmax, m_k, m_nn, points));

    // save the last computed number of particles
    m_Np = Np;
    }

void TransOrderParameter::computePy(trajectory::Box& box,
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

void export_TransOrderParameter()
    {
    class_<TransOrderParameter>("TransOrderParameter", init<float>())
        .def(init<float, float>())
        .def(init<float, float, unsigned int>())
        .def("getBox", &TransOrderParameter::getBox, return_internal_reference<>())
        .def("compute", &TransOrderParameter::computePy)
        .def("getDr", &TransOrderParameter::getDrPy)
        ;
    }

}; }; // end namespace freud::order


