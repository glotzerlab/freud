#ifndef _MATCH_ENV_H__
#define _MATCH_ENV_H__

#include <boost/shared_array.hpp>
//#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include <vector>
#include <set>

#include "Cluster.h"
#include "LinkCell.h"

#include "trajectory.h"
#include <stdexcept>
#include <complex>
#include <map>
#include <algorithm>



namespace freud { namespace order {
//! Clusters particles according to whether their local environments match or not, according to various shape matching metrics

struct Environment
    {
    //! Constructor. Builds an environment indexed by ind
    Environment(unsigned int i) : ind(i), vecs(0);
    //! Is the set of vectors defined by v2 similar to this environment?
    bool isSimilar(std::vector< vec3<float> > v2)
        {
        int blah=0;
        }
    //! Assimilate the set of vectors v2 (INDEXED PROPERLY) into this environment
    void assimilate(std::vector< vec3<float> > v2)
        {
        int blah=0;
        }
    //! Add a vector to define the local environment
    void addVec(vec3<float> vec)
        {
        vecs.push_back(vec);
        }

    unsigned int ind;
    std::vector<vec3<float> > vecs;
    }

class MatchEnv
    {
    public:
        //! Constructor
        /**Constructor for Match-Environment analysis class.  After creation, call compute to calculate clusters grouped by matching environment.  Use accessor functions to retrieve data.
        @param rmax Cutoff radius for cell list and clustering algorithm.  Values near first minimum of the rdf are recommended.
        **/
        MatchEnv(float rmax);

        //! Construct and return a local environment surrounding a particle indexed by i
        Environment MatchEnv::buildEnv(const vec3<float> *points, unsigned int i);

        //! Determine clusters of particles with matching environments
        void compute(const vec3<float> *points, const trajectory::Box& box, unsigned int Np);

        //! Get a reference to the particles, indexed into clusters according to their matching local environments
        boost::shared_array<unsigned int> getClusters()
            {
            return m_env_index;
            }

        //! Returns the set of vectors defining the environment indexed by i
        std::vector< vec3<float> > getEnvironment(unsigned int i)
            {
            const Environment e& = m_env[i];
            return e.vecs;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

    private:
        float m_rmax;               //!< Maximum cutoff radius at which to determine local environment
        locality::LinkCell m_lc;    //!< LinkCell to bin particles for the computation of local environments
        unsigned int m_Np;          //!< Last number of points computed

        boost::shared_array<unsigned int> m_env_index;              //!< Cluster index determined for each particle
        std::vector<Environment> m_env;                             //!< Vector of all local environments
    };

}; }; // end namespace freud::match_env

#endif // _MATCH_ENV_H__
