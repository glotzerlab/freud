## \package freud.locality
#
# Methods and data structures computing properties that at local in space.
#

# bring related c++ classes into the locality module
from _freud import LinkCell
from _freud import IteratorLinkCell


# fixup IteratorLinkCell with an __iter__ method
def iterator_link_cell_iter(self):
    return self
IteratorLinkCell.__iter__ = iterator_link_cell_iter;
