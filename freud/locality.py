## \package freud.locality
#
# Methods and data structures computing properties that at local in space.
#

# bring related c++ classes into the locality module
from _freud import LinkCell
from _freud import IteratorLinkCell
from _freud import NearestNeighbors

# fixup IteratorLinkCell with an __iter__ method
def iterator_link_cell_iter(self):
    return self
IteratorLinkCell.__iter__ = iterator_link_cell_iter;

# fixup IteratorNeighborList with an __iter__ method
# def iterator_neighbor_list_iter(self):
#     return self
# IteratorNeighborList.__iter__ = iterator_neighbor_list_iter;
