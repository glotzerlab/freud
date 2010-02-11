#include "trajectory.h"

#ifndef _LINKCELL_H__
#define _LINKCELL_H__

//! Computes a cell id for each particle and a link cell data structure for iterating through it
/*! For simplicity in only needing a small number of arrays, the link cell algorithm is used to generate and store 
    the cell list data for particles.
    
    Cells are given a nominal minimum width \a cell_width. Each dimension of the box is split into an integer number of
    cells no smaller than \a cell_width wide in that dimension. The actual number of cells along each dimension is
    stored in an Index3D which is also used to compute the cell index from (i,j,k). 
    
    The cell coordinate (i,j,k) itself is computed like so:
    \code
    scalex = 1.0
    i = floorf(x / 
    \endcode
*/


//! Exports all classes in this file to python
void export_LinkCell();

#endif // _TRAJECTORY_H__
