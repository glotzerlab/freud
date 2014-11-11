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

class NNeighbors:
    ## Initialize NearestNeighbors
    # \param box The simulation box
    # \param rmax The maximum distance to search for nearest neighbors
    # \param n The number of nearest neighbors to find
    def __init__(self,rmax,n):
        super(NNeighbors, self).__init__()
        self.rmax = rmax
        self.n = int(n)
        self.handle = NearestNeighbors(self.rmax, self.n)
        self.neighborList = None
        self.RsqList = None

    # change any parameters
    def update(box=None,
               rmax=None,
               n=None):
        if box is not None:
            self.box = box
        if rmax is not None:
            self.rmax = rmax
        if n is not None:
            self.n = n
        if not ((box is None) and (rmax is None) and (n is None)):
            self.handle = NearestNeighbors(self.rmax, self.n)

    # find nearest neighbors
    def compute(self,
                box,
                ref_pos,
                pos):
        self.box = box
        self.handle.compute(self.box,ref_pos,pos)
        self.rmax = self.handle.getRMax()
        self.neighborList = self.handle.getNeighborList()
        self.neighborList = self.neighborList.reshape(shape=(len(ref_pos), self.n))
        self.RsqList = self.handle.getRsqList()
        self.RsqList = self.RsqList.reshape(shape=(len(ref_pos), self.n))

    # return the nearest neighbors of point idx
    def neighbors(self,
                  idx):
        if self.neighborList is not None:
            return self.neighborList[idx]
        else:
            try:
                return self.handle.getNeighbors(idx)
            except:
                raise RuntimeError("neighbors have not been calculated")

    # get the distance between point idx and its nearest neighbors
    def rsq(self,
            idx):
        if self.RsqList is not None:
            return self.RsqList[idx]
        else:
            try:
                return self.handle.getRsq(idx)
            except:
                raise RuntimeError("neighbors have not been calculated")

