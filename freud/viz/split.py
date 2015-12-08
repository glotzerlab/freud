import numpy
import _freud

def split(split_pos, split_ang, pos, ang, centers):
    split_pos = numpy.array(split_pos, dtype=numpy.float32)
    split_ang = numpy.array(split_ang, dtype=numpy.float32)
    pos = numpy.array(pos, dtype=numpy.float32)
    ang = numpy.array(ang, dtype=numpy.float32)
    centers = numpy.array(centers, dtype=numpy.float32)
    _freud.split(split_pos, split_ang, pos, ang, centers)
    return split_pos, split_ang
