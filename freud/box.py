import warnings;
from collections import namedtuple

import numpy as np

from ._freud import Box as _Box;


class Box(_Box):

    @property
    def Lx(self):
        return self.getLx()

    @Lx.setter
    def Lx(self, value):
        self.setL([value, self.Ly, self.Lz])
        return value

    @property
    def Ly(self):
        return self.getLy()

    @Ly.setter
    def Ly(self, value):
        self.setL([self.Lx, value, self.Lz])
        return value

    @property
    def Lz(self):
        return self.getLz()

    @Lz.setter
    def Lz(self, value):
        self.setL([self.Lx, self.Ly, value])
        return value

    @property
    def xy(self):
        return self.getTiltFactorXY()

    @property
    def xz(self):
        return self.getTiltFactorXZ()

    @property
    def yz(self):
        return self.getTiltFactorYZ()

    @property
    def dimensions(self):
        return 2 if self.is2D() else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self.set2D(value == 2)

    def to_dict(self):
        return {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'Lz': self.Lz,
            'xy': self.xy,
            'xz': self.xz,
            'yz': self.yz,
            'dimensions': self.dimensions}

    def to_tuple(self):
        """Returns the box as named tuple."""
        tuple_type = namedtuple('BoxTuple', ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'])
        return tuple_type(Lx=self.Lx, Ly=self.Ly, Lz=self.Lz, xy=self.xy, xz=self.xz, yz=self.yz)

    def to_matrix(self):
        """Returns the box matrix (3x3)."""
        return [[self.Lx, self.xy * self.Ly, self.xz * self.Lz],
                [0, self.Ly, self.yz * self.Lz],
                [0, 0, self.Lz]]


    def __str__(self):
        return "{cls}(Lx={Lx}, Ly={Ly}, Lz={Lz}, xy={xy}, xz={xz}, yz={yz}, dimensions={dimensions})".format(
            cls=type(self).__name__, ** self.to_dict())

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_box(cls, box):
        "Initialize a box instance from another box instance."
        dimensions = getattr(box, 'dimensions', 3)
        return cls(Lx=box.Lx, Ly=box.Ly, Lz=box.Lz, xy=box.xy, xz=box.xz, yz=box.yz, is2D=dimensions==2)

    @classmethod
    def from_matrix(cls, boxMatrix, dimensions=None):
        """Initialize a box instance from a box matrix.

        For more information and the source for this code,
        see: http://hoomd-blue.readthedocs.io/en/stable/box.html
        """
        boxMatrix = np.asarray(boxMatrix, dtype=np.float32)
        v0 = boxMatrix[:,0]
        v1 = boxMatrix[:,1]
        v2 = boxMatrix[:,2]
        Lx = np.sqrt(np.dot(v0, v0))
        a2x = np.dot(v0, v1) / Lx
        Ly = np.sqrt(np.dot(v1,v1) - a2x*a2x)
        xy = a2x / Ly
        v0xv1 = np.cross(v0, v1)
        v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
        Lz = np.dot(v2, v0xv1) / v0xv1mag
        a3x = np.dot(v0, v2) / Lx
        xz = a3x / Lz
        yz = (np.dot(v1,v2) - a2x*a3x) / (Ly*Lz)
        if dimensions is None:
            dimensions = 2 if Lz == 0 else 3
        return cls(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, is2D=dimensions==2)

    @classmethod
    def cube(cls, L):
        """Construct a cubic box.

        :param L: The edge length
        :type L: float
        """
        return cls(Lx=L, Ly=L, Lz=L, xy=0, xz=0, yz=0, is2D=False)

    @classmethod
    def square(cls, L):
        """Construct a 2-dimensional box with equal lengths.

        :param L: The edge length
        :type L: float
        """
        return cls(Lx=L, Ly=L, Lz=0, xy=0, xz=0, yz=0, is2D=True)
