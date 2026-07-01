
import numpy as np
import pytest
import matplotlib.pyplot as plt
import freud 
import warnings 
import io
from importlib.util import find_spec


class TestPlotting:
    
    def test_bar_plot(self):
        x = np.linspace(0, 14, num=15)
        height = np.random.rand(15)
        title = "Test!"
        xlabel = "bins"
        ylabel = "height"

        #Test with default plot settings
        bp1 = freud.plot.bar_plot(x, height)
        assert bp1.xaxis and bp1.yaxis != None
        assert len(bp1.get_title()) == 0
        assert len(bp1.get_xlabel()) == 0
        assert len(bp1.get_ylabel()) == 0
        assert bp1.get_xticks().all() == x.all()
        assert len(bp1.get_xticklabels()) == len(x)
        
        plt.savefig("plot.png")
        
        #Test with ax already created
        fig = plt.figure()
        ax = fig.subplots()
        
        bp2 = freud.plot.bar_plot(x, height, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
        assert bp2.xaxis and bp2.yaxis != None
        assert bp2.get_title() == title
        assert bp2.get_xlabel() == xlabel
        assert bp2.get_ylabel() == ylabel
        assert bp2.get_xticks().all() == x.all()
        assert len(bp2.get_xticklabels()) == len(x)
        
        plt.savefig("plot2.png")

        #Test with no default plot settings
        bp3 = freud.plot.bar_plot(x, height, title=title, xlabel=xlabel, ylabel=ylabel)
        assert bp3.xaxis and bp3.yaxis != None
        assert bp3.get_title() == title
        assert bp3.get_xlabel() == xlabel
        assert bp3.get_ylabel() == ylabel
        assert bp3.get_xticks().all() == x.all()
        assert len(bp3.get_xticklabels()) == len(x)
        plt.savefig("plot3.png")


test = TestPlotting()
test.test_bar_plot()
        

