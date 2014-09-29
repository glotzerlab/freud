import re
try:
    import OpenGL
    from PySide import QtCore, QtGui, QtOpenGL
except ImportError:
    logger.warning('Either PySide or pyopengl is not available, aborting rt initialization');
    raise ImportWarning('PySide or pyopengl not available')

class Options(QtGui.QWidget):
    # which options do we need?
    # well, options need to be based on type...
    def __init__(self,
                 phiList=None,
                 runList=None,
                 rdf=True,
                 ocf=True,
                 bocf=True,
                 angleSymmetry=None,
                 rMax=None,
                 ocfrMax=None,
                 outline=None,
                 *args,
                 **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        # create combo boxes to select density and run number
        self.phiComboBox = QtGui.QComboBox()
        for item in phiList:
            self.phiComboBox.addItem(item)
        self.runComboBox = QtGui.QComboBox()
        for item in runList:
            self.runComboBox.addItem(item)
        # add combo boxes
        row1 = QtGui.QHBoxLayout()
        row1.addWidget(QtGui.QLabel('Select phi:'))
        row1.addWidget(self.phiComboBox)
        row1.addWidget(QtGui.QLabel('Select run:'))
        row1.addWidget(self.runComboBox)

        # coloring options
        self.coloringComboBox = QtGui.QComboBox()
        self.coloringComboBox.addItem('normal')
        self.coloringComboBox.addItem('angle')
        self.coloringComboBox.addItem('psi angle')
        self.coloringComboBox.addItem('psi magnitude')

        # create boxes to select outline thickness, angle symmetry, and k-value
        self.outlineLineEdit = QtGui.QLineEdit("{}".format(outline))
        self.angleLineEdit = QtGui.QLineEdit("{}".format(angleSymmetry))
        self.kLineEdit = QtGui.QLineEdit("{}".format(kValue))

        # add boxes
        row2 = QtGui.QHBoxLayout()
        row2.addWidget(QtGui.QLabel('Color by:'))
        row2.addWidget(self.coloringComboBox)
        row2.addWidget(QtGui.QLabel('Outline thickness:'))
        row2.addWidget(self.outlineLineEdit)
        row2.addWidget(QtGui.QLabel('Angle symmetry:'))
        row2.addWidget(self.angleLineEdit)
        row2.addWidget(QtGui.QLabel('k-value:'))
        row2.addWidget(self.kLineEdit)

        self.rdfLineEdit = QtGui.QLineEdit("{}".format(rMax))
        self.ocfLineEdit = QtGui.QLineEdit("{}".format(ocfrMax))
        row3 = QtGui.QHBoxLayout()
        row4 = QtGui.QHBoxLayout()
        row3.addWidget(QtGui.QLabel('rMax:'))
        row3.addWidget(self.rdfLineEdit)
        row3.addWidget(QtGui.QLabel('ocfrMax:'))
        row3.addWidget(self.ocfLineEdit)
        # create a checkbox to turn on visualization of plots
        row4 = QtGui.QHBoxLayout()
        # create a force reload option
        self.forceReloadPushButton = QtGui.QPushButton("Force Reload")
        row4.addWidget(self.forceReloadPushButton)
        self.recordMoviePushButton = QtGui.QPushButton("Record Movie")
        row4.addWidget(self.recordMoviePushButton)
        self.dumpPlotsPushButton = QtGui.QPushButton("Dump Plots")
        row4.addWidget(self.dumpPlotsPushButton)

        row5 = QtGui.QHBoxLayout()
        self.rdfCheckBox = QtGui.QCheckBox()
        if rdf == True:
            self.rdfCheckBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.rdfCheckBox.setCheckState(QtCore.Qt.Unchecked)
        self.ocfCheckBox = QtGui.QCheckBox()
        self.bocfCheckBox = QtGui.QCheckBox()
        if ocf == True:
            self.ocfCheckBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.ocffCheckBox.setCheckState(QtCore.Qt.Unchecked)
        if bocf == True:
            self.bocfCheckBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.bocffCheckBox.setCheckState(QtCore.Qt.Unchecked)
        row5.addWidget(QtGui.QLabel("use rdf"))
        row5.addWidget(self.rdfCheckBox)
        row5.addWidget(QtGui.QLabel("use ocf"))
        row5.addWidget(self.ocfCheckBox)
        row5.addWidget(QtGui.QLabel("use bocf"))
        row5.addWidget(self.bocfCheckBox)

        self.widthLineEdit = QtGui.QLineEdit("{}".format(1024))
        self.heightLineEdit = QtGui.QLineEdit("{}".format(800))
        row6 = QtGui.QHBoxLayout()
        row6.addWidget(QtGui.QLabel('Movie Width:'))
        row6.addWidget(self.widthLineEdit)
        row6.addWidget(QtGui.QLabel('Movie Height:'))
        row6.addWidget(self.heightLineEdit)


        layout = QtGui.QVBoxLayout()
        if compsim == True:
            layout.addLayout(row0)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        layout.addLayout(row4)
        layout.addLayout(row5)
        layout.addLayout(row6)

        self.setLayout(layout)