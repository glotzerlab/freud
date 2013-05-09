#include "HexOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>
#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;
using namespace tbb;

namespace freud { namespace order {
    
HexOrderParameter::HexOrderParameter(const trajectory::Box& box, float rmax)
    :m_box(box), m_rmax(rmax), m_lc(box, rmax)
    {
    }

class ComputeHexOrderParameter
    {
    private:
        const trajectory::Box& m_box;
        const float m_rmax;
        const locality::LinkCell& m_lc;
        const float3 *m_points;
        std::complex<float> *m_psi_array;
    public:
        ComputeHexOrderParameter(std::complex<float> *psi_array,
                                 const trajectory::Box& box,
                                 const float rmax,
                                 const locality::LinkCell& lc,
                                 const float3 *points)
            : m_box(box), m_rmax(rmax), m_lc(lc), m_points(points), m_psi_array(psi_array)
            {
            }
        
        void operator()( const blocked_range<size_t>& r ) const
            {
            const float3 *points = m_points;
            float rmaxsq = m_rmax * m_rmax;
            
            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                //get cell point is in
                float3 ref = points[i];
                unsigned int ref_cell = m_lc.getCell(ref);
                unsigned int num_adjacent = 0;
                
                //loop over neighboring cells
                const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    unsigned int neigh_cell = neigh_cells[neigh_idx];
                    
                    //iterate over particles in cell
                    locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                        {
                    /*for (unsigned int k = i; k < i+6; k++)
                        {
                        unsigned int j = k % (1024*1024);*/
                        //compute r between the two particles
                        float dx = float(ref.x - points[j].x);
                        float dy = float(ref.y - points[j].y);
                        float dz = float(ref.z - points[j].z);
                        float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                        
                        float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        if (rsq < rmaxsq && rsq > 1e-6)
                            {
                            //compute psi for neighboring particle(only constructed for 2d)
                            float psi_ij = atan2f(delta.y, delta.x);
                            m_psi_array[i] += exp(complex<float>(0,6*psi_ij));
                            num_adjacent++;
                            }
                        }
                    }
                
                // Don't divide by zero if the particle has no neighbors (this leaves psi at 0)
                if(num_adjacent)
                    m_psi_array[i] /= complex<float>(num_adjacent);
                }
            }
    };

void HexOrderParameter::compute(const float3 *points, unsigned int Np)
    {
    tick_count t0 = tick_count::now();
    m_lc.computeCellList(points,Np);
    tick_count t1 = tick_count::now();
    cout << "lc build time: " << (t1-t0).seconds() << endl;


    t0 = tick_count::now();    
    m_Np = Np;
    float rmaxsq = m_rmax * m_rmax;
    m_psi_array = boost::shared_array<complex<float> >(new complex<float> [Np]);
    memset((void*)m_psi_array.get(), 0, sizeof(complex<float>)*Np);
    
    t1 = tick_count::now();
    cout << "allocate time: " << (t1-t0).seconds() << endl;


    float base_time = 0;
    
    for (unsigned int nthreads=1; nthreads <= 8; nthreads++)
        {
        task_scheduler_init init(nthreads);
        static affinity_partitioner ap;

        parallel_for(blocked_range<size_t>(0,Np), ComputeHexOrderParameter(m_psi_array.get(), m_box, m_rmax, m_lc, points), ap);

        t0 = tick_count::now();
                
        parallel_for(blocked_range<size_t>(0,Np), ComputeHexOrderParameter(m_psi_array.get(), m_box, m_rmax, m_lc, points), ap);
        
        t1 = tick_count::now();
        float t = (t1-t0).seconds();
        if (nthreads==1)
            {
            cout << "compute time 1: " << t << endl;
            base_time = t;
            }
        else
            {
            cout << "compute time " << nthreads << ": " << t << " speedup=" << base_time/t << endl;
            }
        
        }
    }

void HexOrderParameter::computePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);
        
        // compute the order parameter with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np);
        }
    }
    
void export_HexOrderParameter()
    {
    class_<HexOrderParameter>("HexOrderParameter", init<trajectory::Box&, float>())
        .def("getBox", &HexOrderParameter::getBox, return_internal_reference<>())
        .def("compute", &HexOrderParameter::computePy)
        .def("getPsi", &HexOrderParameter::getPsiPy)
        ;
    }
    
}; }; // end namespace freud::order


