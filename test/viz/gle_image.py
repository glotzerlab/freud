from __future__ import division, print_function
import math
import random
import numpy

from freud import viz

if __name__ == '__main__':
    
    data = numpy.ones(shape=(256,256), dtype=numpy.float32);
    data[:,:] = numpy.linspace(0,1,256);
    
    map = viz.colormap.grayscale(data);
    img = viz.primitive.Image(position=(0,0), size=(10,10), data=map);
    
    group = viz.base.Group(primitives=[img]);
    cam = viz.base.Camera(position=(5,5,1), look_at=(5,5,0), up=(0,1,0), aspect=1, height=12);
    scene = viz.base.Scene(camera=cam, groups=[group]);
    
    writer = viz.export.gle.WriteGLE()
    writer.write(open('gle_image.gle', 'wb'), scene);
