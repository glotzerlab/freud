import re
import json
import numpy

class Params(object):
    def __init__(self,
                 paramFile=None,
                 analysisFile=None):
        super(Params, self).__init__()
        if paramFile is not None:
            self.params = json.load(open(paramFile, "r"))
            # Handle old params files
            try:
                self.simType = str(self.params["simType"])
            except KeyError:
                self.simType = str(self.params["sim_type"])
            try:
                self.waveForm = str(self.params["waveForm"])
            except KeyError:
                self.waveForm = "triangle"
            try:
                self.baseShape = int(self.params["baseShape"])
            except KeyError:
                self.baseShape = int(self.params["base_shape"])
            try:
                self.shapeType = str(self.params["shapeType"])
            except KeyError:
                self.shapeType = str(self.params["shape_type"])
            try:
                self.fileName = str(self.params["fileName"])
            except KeyError:
                self.fileName = str(self.params["fname"])
            self.verts = {}
            self.r = float(self.params["r"])
            try:
                self.nRuns = int(self.params["nRuns"])
            except KeyError:
                self.nRuns = int(self.params["nruns"])
            # m is the number of particle (pairs) per side
            m = int(self.params["m"])
            if self.shapeType == "base":
                self.numParticles = m * m
            else:
                self.numParticles = 2 * m * m
            # Create boolean value for simpler changes
            self.myEnumerate = True
            self.compsim = False
            self.trapWave = False
            # create boolean value for movie recording
            self.isRecording = False
            self.movieWidth = 1024
            self.movieHeight = 800
            if (bool(re.match("trap", self.waveForm))):
                self.trapWave = True

            if (bool(re.match("comp", self.shapeType)) or bool(re.match(".comp", self.shapeType))):
                self.compsim = True

            # if (bool(re.match("base", self.shapeType))) or ((self.baseShape == 4) and (bool(re.match("esplit", self.shapeType)))):
            if (bool(re.match("base", self.shapeType))):
                self.usePairing = False

            if self.compsim == True:
                # Again, support older files
                if self.trapWave == True:
                    etaMin = self.params['etaMin']
                    etaMax = self.params['etaMax']
                    etaNum = self.params["etaNum"]
                    etaArray = numpy.linspace(etaMin, etaMax, etaNum)
                    self.etaList = ["{:.5}".format(x) for x in etaArray]
                try:
                    nkMin = self.params['nkMin']
                    nkMax = self.params['nkMax']
                except KeyError:
                    try:
                        nkMin = self.params['nk-min']
                        nkMax = self.params['nk-max']
                    except KeyError:
                        nkMin = self.params['wv-min']
                        nkMax = self.params['wv-max']
                try:
                    nkNum = float(self.params['nkNum'])
                except KeyError:
                    nkNum = nkMax - nkMin + 1
                try:
                    aMin = self.params['aMin']
                    aMax = self.params['aMax']
                    aNum = self.params["aNum"]
                except KeyError:
                    aMin = self.params['A-min']
                    aMax = self.params['A-max']
                    aNum = self.params["A-num"]
                nkArray = numpy.linspace(nkMin, nkMax, nkNum)
                self.nkList = ["{:.2}".format(x) for x in nkArray]
                AArray = numpy.linspace(aMin, aMax, aNum)
                self.aList = ["{:.5}".format(x) for x in AArray]
            try:
                phiMin = self.params['phiMin']
                phiMax = self.params['phiMax']
                phiNum = self.params["phiNum"]
            except KeyError:
                phiMin = self.params['phi-min']
                phiMax = self.params['phi-max']
                phiNum = self.params["phi-num"]
            phiArray = numpy.linspace(phiMin, phiMax, phiNum)
            self.phiList = ["{:.5}".format(x) for x in phiArray]
            runArray = []
            for i in range(self.nRuns):
                runArray.append(i)
            self.runList = ["{}".format(x) for x in runArray]
            # set up options:
            if self.compsim == True:
                if self.trapWave != True:
                    self.etaList = None
            else:
                self.etaList = None
                self.nkList = None
                self.aList = None
        # now for the analysis file
        if analysisFile is not None:
            try:
                self.params = json.load(open(analysisFile, "r"))
                # Handle older analysis files
                try:
                    self.angleSymmetry = float(self.params['angleSymmetry'])
                except:
                    self.angleSymmetry = 1.0
                try:
                    self.outline = float(self.params['outline'])
                except:
                    self.outline = 0.05
                try:
                    self.rMax = float(self.params['rMax'])
                except KeyError:
                    self.rMax = float(self.params['r_max'])
                self.dr = float(self.params['dr'])
                try:
                    self.ocfrMax = float(self.params['ocfrMax'])
                except KeyError:
                    try:
                        self.ocfrMax = float(self.params['wrMax'])
                    except KeyError:
                        self.ocfrMax = float(self.params["wr_max"])
                try:
                    self.ocfdr = float(self.params["ocfdr"])
                except KeyError:
                    self.ocfdr = float(self.params["wdr"])
                try:
                    self.ldMax = float(self.params['ldMax'])
                except:
                    self.ldMax = 3.0
                try:
                    self.kMax = float(self.params["kMax"])
                except:
                    self.kMax = 1.5
                try:
                    self.k = float(self.params["k"])
                except:
                    self.k = 2.0
                try:
                    self.xMax = float(self.params['xMax'])
                except:
                    self.xMax = 3.0
                try:
                    self.dx = float(self.params['dx'])
                except:
                    self.dx = 0.05
                try:
                    self.yMax = float(self.params['yMax'])
                except:
                    self.yMax = 3.0
                try:
                    self.dy = float(self.params['dy'])
                except:
                    self.dy = 0.05
                try:
                    self.pmfMax = float(self.params['pmfMax'])
                except:
                    self.pmfMax = 5.0
                try:
                    self.dCut = float(self.params['dCut'])
                except KeyError:
                    self.dCut = float(self.params["d_cut"])
                try:
                    self.cDotTol = float(self.params["cDotTol"])
                except KeyError:
                    self.cDotTol = float(self.params["c_dot_tol"])
                self.useRDF = bool(self.params["rdf"] == "True")
                try:
                    self.useOCF = bool(self.params["ocf"] == "True")
                except KeyError:
                    self.useOCF = bool(self.params["wrdf"] == "True")
                self.usePMF = bool(self.params["pmf"] == "True")
                self.usePairing = bool(self.params["pairing"] == "True")
                self.useTheta = bool(self.params["theta"] == "True")

            except IOError:
                # Just use some default values and not break
                print("could not find analysis.json; using default values")
                self.angleSymmetry = 1.0
                self.outline = 1.0
                self.rMax = 3.0
                self.dr = 0.01
                self.ocfrMax = 3.0
                self.ocfdr = 0.01
                self.ldMax = 3.0
                self.kMax = 1.5
                self.k = 2.0
                self.xMax = 3.0
                self.dx = 0.05
                self.yMax = 3.0
                self.dy = 0.05
                self.pmfMax = 5.0
                self.dCut = 1.0
                self.cDotTol = 0.1
                self.useRDF = True
                self.useOCF = True
                self.usePMF = True
                self.usePairing = True
                self.useTheta = True