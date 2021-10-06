#ifndef STATIC_STRUCTURE_FACTOR_H
#define STATIC_STRUCTURE_FACTOR_H

namespace freud { namespace diffraction {

class StaticStructureFactor
{
protected:
    using StructureFactorHistogram = util::Histogram<float>;

    StaticStructureFactor() = default;

    StaticStructureFactor(unsigned int bins, float k_max, float k_min = 0);

public:
    virtual ~StaticStructureFactor() = default;

    virtual void accumulate(const freud::locality::NeighborQuery* neighbor_query,
                            const vec3<float>* query_points,
                            unsigned int n_query_points,
                            unsigned int n_total) = 0;

    virtual void reset() = 0;

    /*
    template<typename U> U& reduceAndReturn(U& thing_to_return)
    {
        if (m_reduce)
        {
            reduce();
        }
        m_reduce = false;
        return thing_to_return;
    }
    */

    virtual const util::ManagedArray<float>& getStructureFactor() = 0;

    virtual std::vector<float> getBinEdges() const = 0;
    /*
    {
        return m_structure_factor.getBinEdges()[0];
    }
    */

    virtual std::vector<float> getBinCenters() const = 0;
    /*
    {
        return m_structure_factor.getBinCenters()[0];
    }
    */

    /*
    float getMinValidK() const
    {
        return m_min_valid_k;
    }
    */

protected:
    // histogram of values for the structure factor
    //StructureFactorHistogram m_structure_factor;
    //StructureFactorHistogram::ThreadLocalHistogram m_local_structure_factor;

    //bool m_reduce {true};  //! whether to reduce local histograms
    //float m_min_valid_k { std::numeric_limits<float>::infinity() }; //! min valid k-vector


};

}; };

#endif // STATIC_STRUCTURE_FACTOR_H
