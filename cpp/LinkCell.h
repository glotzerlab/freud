#include <boost/shared_array.hpp>

#include "trajectory.h"
#include "HOOMDMath.h"
#include "Index1D.h"

#ifndef _LINKCELL_H__
#define _LINKCELL_H__

const unsigned int LINK_CELL_TERMINATOR = 0xffffffff;

//! Computes a cell id for each particle and a link cell data structure for iterating through it
/*! For simplicity in only needing a small number of arrays, the link cell algorithm is used to generate and store 
    the cell list data for particles.
    
    Cells are given a nominal minimum width \a cell_width. Each dimension of the box is split into an integer number of
    cells no smaller than \a cell_width wide in that dimension. The actual number of cells along each dimension is
    stored in an Index3D which is also used to compute the cell index from (i,j,k).
    
    The cell coordinate (i,j,k) itself is computed like so:
    \code
    i = floorf((x + Lx/2) / w) % Nw
    \endcode
    and so on for j,k (y,z). Call getCellCoord to do this computation for an arbitrary point.
    
    <b>Data structures:</b><br>
    
*/
class LinkCell
    {
    public:
        //! Constructor
        LinkCell(const Box& box, float cell_width);
        
        //! Get the simulation box
        const Box& getBox() const
            {
            return m_box;
            }
        
        //! Get the cell indexer
        const Index3D& getCellIndexer() const
            {
            return m_cell_index;
            }
        
        //! Get the number of cells
        unsigned int getNumCells() const
            {
            return m_cell_index.getNumElements();
            }
        
        //! Compute the cell id for a given position
        unsigned int getCell(float x, float y, float z) const
            {
            uint3 c = getCellCoord(x,y,z);
            return m_cell_index(c.x, c.y, c.z);
            }
            
        //! Compute cell coordinates for a given position
        uint3 getCellCoord(float x, float y, float z) const
            {
            uint3 c;
            c.x = floorf((x + m_box.getLx()/2.0f) * (float(m_cell_index.getW())/m_box.getLx()));
            c.x %= m_cell_index.getW();
            c.y = floorf((y + m_box.getLy()/2.0f) * (float(m_cell_index.getH())/m_box.getLy()));
            c.y %= m_cell_index.getH();
            c.z = floorf((z + m_box.getLz()/2.0f) * (float(m_cell_index.getD())/m_box.getLz()));
            c.z %= m_cell_index.getD();
            return c;
            }
        
        //! Compute the cell list
        void computeCellList(float *x, float *y, float *z, unsigned int Np);
    private:
        Box m_box;              //!< Simulation box the particles belong in
        Index3D m_cell_index;   //!< Indexer to compute cell indices
        
        boost::shared_array<unsigned int> m_cell_list;    //!< The cell list last computed
    };

//! Exports all classes in this file to python
void export_LinkCell();

#endif // _TRAJECTORY_H__
