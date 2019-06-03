#ifndef NDHISTOGRAM_H
#define NDHISTOGRAM_H

#include "Box.h"
#include "NeighborList.h"

namespace freud { namespace util {
class NdHistogram 
    {
    public:
        NdHistogram() {}

        virtual ~NdHistogram() {}

        const box::Box& getBox() const
            {
            return m_box;
            }

        unsigned int getNBins() const
            {
            return m_nbins;
            }

        std::shared_ptr<float> getR()
            {
                return m_r_array;
            }

        void reduce();

        template <typename Func>
        void accumulateGeneral(box::Box& box, 
                        const freud::locality::NeighborList *nlist,
                        const vec3<float> *ref_points,
                        unsigned int n_ref,
                        const vec3<float> *points,
                        unsigned int Np, Func fn);

        void reset();

    private:
        box::Box m_box;                //!< Simulation box where the particles belong
        float m_rmax;                  //!< Maximum r at which to compute g(r)
        float m_dr;                    //!< Step size for r in the computation
        unsigned int m_nbins;          //!< Number of r bins to compute g(r) over
        unsigned int m_n_ref;          //!< number of reference particles
        unsigned int m_Np;             //!< number of check particles
        unsigned int m_frame_counter;  //!< number of frames calc'd
        bool m_reduce;                 //!< Whether arrays need to be reduced across threads

        std::shared_ptr<T> m_rdf_array;             //!< rdf array computed
        std::shared_ptr<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        std::shared_ptr<float> m_r_array;           //!< array of r values where the rdf is computed
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
        tbb::enumerable_thread_specific<T *> m_local_rdf_array;
    };

}; }; // end namespace freud::util

#endif // NDHISTOGRAM_H
