## \package freud.locality
#
# Computes locality information on sets of points
#
# The following classes are imported into locality from C++:
#  - LinkCell
#  - IteratorLinkCell
#  - Cluster

# bring related c++ classes into the locality module
from _freud import LinkCell
from _freud import IteratorLinkCell
from _freud import Cluster


# fixup IteratorLinkCell with an __iter__ method
def iterator_link_cell_iter(self):
    return self
IteratorLinkCell.__iter__ = iterator_link_cell_iter;
