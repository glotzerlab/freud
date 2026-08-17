import numpy as np
import pytest
import matplotlib.pyplot as plt
import freud 
import warnings 
import io
from importlib.util import find_spec


class TestPlotting:
    
    def test_box_plot(self):
        box = freud.box.Box.cube(6)
        points = np.random.randint(1, high = 3, size=(3,100))
        
        title = "Test!"
        image = [0, 0, 0]

        #Test with default plot settings
        bp1 = freud.plot.box_plot(box=box)
        bp1.plot(points[0], points[1], points[2], marker='.')
        assert bp1.xaxis and bp1.yaxis != None and bp1.zaxis != None
        assert len(bp1.get_title()) == 0
        assert bp1.get_xlabel() == "$x$"
        assert bp1.get_ylabel() == "$y$"
        assert bp1.get_zlabel() == "$z$"
        
        #plt.savefig("plot.png")
        
        #Test with ax already created
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        bp2 = freud.plot.box_plot(box=box, title=title, ax=ax, image=image)
        bp2.plot(points[0], points[1], points[2], marker='.')
        assert bp2.xaxis and bp2.yaxis != None and bp2.zaxis != None
        assert bp2.get_title() == title
        assert bp2.get_xlabel() == "$x$"
        assert bp2.get_ylabel() == "$y$"
        assert bp2.get_zlabel() == "$z$"
        
        
        #plt.savefig("plot2.png")
        
        #Test with no default plot settings (except ax)
        bp3 = freud.plot.box_plot(box=box, title=title, image=image)
        bp3.plot(points[0], points[1], points[2], marker='.')
        assert bp3.xaxis and bp3.yaxis != None and bp3.zaxis != None
        assert bp3.get_title() == title
        assert bp3.get_xlabel() == "$x$"
        assert bp3.get_ylabel() == "$y$"
        assert bp3.get_zlabel() == "$z$"
       
        #plt.savefig("plot3.png")

        #Test with box and ax in 2D and no default plotting settings
        box2 = freud.box.Box(Lx=4, Ly=6, is2D=True)
        fig = plt.figure()
        ax = fig.subplots()

        bp4 = freud.plot.box_plot(box=box2, title=title, ax=ax, image=image)
        bp4.plot(points[0], points[1], marker='.')
        assert bp4.xaxis and bp4.yaxis != None
        assert bp4.get_title() == title
        assert bp4.get_xlabel() == "$x$"
        assert bp4.get_ylabel() == "$y$"
        #bp4.set_title(title)

        #plt.savefig("plot4.png")

        #Test with box is 2D and default ax settings
        bp5 = freud.plot.box_plot(box=box2)
        bp5.plot(points[0], points[1], marker='.')
        assert bp5.xaxis and bp5.yaxis != None
        assert len(bp5.get_title()) == 0
        assert bp5.get_xlabel() == "$x$"
        assert bp5.get_ylabel() == "$y$"
        
        #plt.savefig("plot5.png")

        
test = TestPlotting()
test.test_box_plot()