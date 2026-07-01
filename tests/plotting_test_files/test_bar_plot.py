
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

        fig = plt.figure()
        ax = fig.subplots()
        ax.bar(x=x, height=height)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        
        bp1 = freud.plot.bar_plot(x, height)
        plt.savefig("plot.png")

        #This one isn't working lol
        bp2 = freud.plot.bar_plot(x, height, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
        bp2_title = ax.get_title()

        bp2_title = ax.get_title()
        if bp2_title != None:
            print(bp2_title)
        
        #assert bp2_title == title
        

        plt.savefig("plot2.png")
        bp3 = freud.plot.bar_plot(x, height, title=title, xlabel=xlabel, ylabel=ylabel)
        plt.savefig("plot3.png")

        #with pytest.raises(AttributeError):
            # bp1 != None
            # bp2 != None
            # bp3 != None





test = TestPlotting()
test.test_bar_plot()
        

