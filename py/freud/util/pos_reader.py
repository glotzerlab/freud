import os
import sys
import re
import numpy as n
import subprocess
import time
import string

class file:
    """
        object class to read/parse incsim output and/or distill data and position
    """
    def __init__(self, infile=None, datfile=None, posfile=None, only_last=False):
        self.infile = infile
        self.datfile = datfile
        self.posfile = posfile
        self.observables = list()
        self.data = dict()
        self.pos = dict()
        self.steps = list() # list of ints
        self.definitions = []
        self.shape_prefix = ''
        self.particle_list = [] # list of 'Shape' objects with hold particle attributes
        self.object_list = [] # list of bonds and polys (Non-particle objects)
        self.position_list = []
        self.quaternion_list = []
        self.boxMatrix_list = []
        self.frame_list = [] # list of ints

        #a bunch of Injavis environment variables
        self.injavis_params=dict()
        self.injavis_params['translation']=None
        self.injavis_params['center']=None
        self.injavis_params['rotation']=None
        self.injavis_params['zoomFactor']=None
        self.injavis_params['transparent']=None
        self.injavis_params['perspective']=None
        self.injavis_params['boxSides']=None
        self.injavis_params['showEdges']=None
        self.injavis_params['addFog']=None
        self.injavis_params['antiAliasing']=None
        self.injavis_params['cartesian']=None
        if not infile is None:
            self.load(only_last=only_last)
    def load(self, filename=None,only_last=False):
        """
            Read an incsim output file into memory. Invoked automatically if
            infile is specified to the constructor.

            filename may be a valid filename, an open file descriptor, or a
            list of lines as from a file.
        """
        if filename is None:
            filename = self.infile
        if filename is None:
            print("No input file assigned")
        else:
            # I got rid of all this; don't know how it affects functionality

            #if isinstance(filename, file):
            #    f = filename
            #elif isinstance(filename, list):
            #    f = filename
            #else:
            #   f = open(filename, 'rU')
            f = open(filename, 'rU')
            i=0
            isdata=False
            # Need to get rid of appends
            # Need to load directly into numpy arrays
            self.particle_list.append([])
            self.definitions.append(dict())
            self.position_list.append([])
            self.object_list.append([])
            self.quaternion_list.append([])

# I think richmond is correct; I need to read through once to initialize size and stuff first
# Count number of frames, number of particles, etc.

            pbuff=old_pbuff=[]
            pbuff_frame=old_pbuff_frame = 0
            # This is the main loop
            for line in f:
                #i+=1
                if re.match('^//',line):
                    continue
                elif re.match('#\[data\]', line):
                    #print("Found data header on line %i" % i)
                    if len(self.observables) == 0:
                        # Takes the observables and puts in list
                        # Self.data is a dict
                        # probably want to change from list to numpy array
                        self.observables = re.split('\s+', line)[1:-1]
                        for col in self.observables:
                            self.data[col]=list()
                    isdata=True

                    #if the buff is full, empty it
                    # if we reading all the frames
                    if len(pbuff)>0 and not only_last:
                        # Append is slow; change
                        self.frame_list.append(pbuff_frame)
                        for lbuff in pbuff:
                            # I'm sure this is ridiculously slow
                            self.addPosline(lbuff.strip())

                    #otherwise save a copy of this buff in case the
                    #next frame is corrupt
                    # Looks fine
                    else:
                      old_pbuff=list(pbuff)
                      old_pbuff_frame=pbuff_frame

                    #clear the pos buffer
                    pbuff = []

                    continue
                elif re.match('#\[done\]', line):
                    isdata=False
                    continue
                if isdata:
                    # I'm sure this could be faster
                    self.addDataline(line)
                else:
                  if ( re.match("^eof",line) ):
                    try:
                        pbuff_frame = self.steps[len(self.steps)-1]
                    except:
                        i += 1
                        pbuff_frame = i
                  # This could be improved; append is slow
                  pbuff.append(line)

            #did the buffer end full?
            if len(pbuff)>0:
              #are both buffers complete?
              #if so, read the newer one
              if len(pbuff)==len(old_pbuff):
                #final buff is complete, add to data set
                for lbuff in pbuff:
                  self.addPosline(lbuff.strip())
                self.frame_list.append(pbuff_frame)
              elif only_last:
                #last buff was incomplete, read in the old one
                #this isn't necessary if the only_last is false
                #because it has already been read in
                for lbuff in old_pbuff:
                  self.addPosline(lbuff.strip())
                self.frame_list.append(old_pbuff_frame)
            elif only_last:
              #read in the last buffer, this file was
              #truncated in a data section
              for lbuff in old_pbuff:
                self.addPosline(lbuff.strip())
              self.frame_list.append(old_pbuff_frame)
            self.particle_list.pop()
            self.position_list.pop()
            self.object_list.pop()
            self.quaternion_list.pop()
            self.definitions.pop()

    def writePos(self,filename,boxMatrix_list,position_list,quaternion_list,particle_list,injavis_params=None,object_list=None,comment=None):
        if os.path.exists(filename):
            pass
        f_out = open(filename,'w+')
        if not comment is None:
            f_out.write('//%s\n'%(comment))

        if not injavis_params is None:
            for k in injavis_params.keys():
                f_out.write('%s\t%s\n'%(k,injavis_params[k]))
        if not boxMatrix_list is None:
          if not len(boxMatrix_list)==9:
              raise RuntimeError('Box matrix must contain 9 elements')
          #write the box matrix
          f_out.write(string.join(['boxMatrix', string.join(['\t%5.8f'%(b) for b in boxMatrix_list]),'\n']))
        d_list = dict();
        counter=0
        for p in particle_list:
            if (not any([p==k for k in d_list.keys()]) or counter==0):
                d_list[p] = 'shape%d'%(counter)
                counter+=1

        for k in d_list.keys():
            f_out.write('def\t%s\t%s\n'%(d_list[k],k.definition))

        for p,x,q in zip(particle_list,position_list,quaternion_list):
            if q.size==4:
                f_out.write('%s\t%5.8f\t%5.8f\t%5.8f\t%5.8f\t%5.8f\t%5.8f\t%5.8f\n'\
                            %(d_list[p],x[0],x[1],x[2],q[0],q[1],q[2],q[3]))
            elif q.size==0:
                f_out.write('%s\t%5.8f\t%5.8f\t%5.8f\n'%(d_list[p],x[0],x[1],x[2]))
            else:
                raise RuntimeError('Invalid quaternion')
        if not object_list is None:
            for obj in object_list:
                f_out.write('%s\n'%(obj.definition))

        f_out.write('eof')
        f_out.close()
    def dumpData(self, filename=None):
        """
            Dump observables data to a file or STDOUT.
            Note: does not currently check whether newline characters need to be added.
        """
        if filename is None:
            filename = self.datfile
        if filename is None:
            f = sys.stdout
        else:
            f = open(filename, 'w')
        f.write("#[data]\t"+"\t".join(self.observables) + '\n')
        for i in range(0,len(self.steps)):
            line = [str(self.data[col][i]) for col in self.observables]
            f.write("\t".join(line) + '\n')
        f.write('#[done]\n')
    def getFrame(self, frame=None):
        """
            Build Pos file text for a single frame of position information
        """
        output=list()
        length = len(self.frame_list) - 1
        if length < 0:
            # no pos data
            # raise exception of some sort, maybe
            return None
        if frame is None:
            frame= length
        line = 'boxMatrix ' + ' '.join([str(i) for i in self.boxMatrix_list[frame]]) + '\n'
        output.append(line)

        d_list = dict();
        counter=0
        for p in self.particle_list[frame]:
            if (not any([p==k for k in d_list.keys()]) or counter==0):
                d_list[p] = 'shape%d'%(counter)
                counter+=1

        for k in d_list.keys():
            output.append('def\t%s\t%s\n'%(d_list[k],k.definition))

        for p,x,q in zip(self.particle_list[frame],self.position_list[frame],self.quaternion_list[frame]):
            if q.size==4:
                output.append('%s\t%5.8f\t%5.8f\t%5.8f\t%5.8f\t%5.8f\t%5.8f\t%5.8f\n'\
                            %(d_list[p],x[0],x[1],x[2],q[0],q[1],q[2],q[3]))
            elif q.size==0:
                output.append('%s\t%5.8f\t%5.8f\t%5.8f\n'%(d_list[p],x[0],x[1],x[2]))
            else:
                raise RuntimeError('Invalid quaternion')
        output.append('eof\n')
        return output
    def dumpPos(self, filename=None):
        """
            Dump pos data to a file or STDOUT.
        """
        if filename is None:
            filename = self.posfile
        if filename is None:
            f = sys.stdout
        else:
            f = open(filename, 'w')
        for frame in range(0,len(self.frame_list)):
            f.writelines(self.getFrame(frame))

    def addDataline(self, line):
        """
            Load a dictionary of observables information as numpy arrays.
        """
        goodline=True
        try:
            numbers = [float(l) for l in line.split()]
        except:
            print("Bad data line after "+str(self.steps[len(self.steps)-1]))
            goodline=False
        if goodline:
            for col in self.observables:
                i = self.observables.index(col)
                self.data[col].append(numbers[i])
            self.steps.append(int(numbers[0]))
    def getData(self, observables=None, minstep=0):
        """
            Return a dictionary of observables information as numpy arrays.

            If a list of observables is provided, only return the
            specified arrays. If minstep is provided, skip values
            from before the indicated step in the incsim data.

            To check the available observables, look at instance.available
        """
        output = dict()
        data=dict()
        column = list()
        available = self.observables
        if observables is None:
            column = available
        else:
            if not isinstance(observables, list):
                observables = list([ observables ])
            pattern = [ re.compile( observable, re.I ) for observable in available ]
            for obs in observables:
                for p in pattern:
                    m = p.search(obs)
                    if m:
                        column.append(p.pattern)
        for col in column:
            data[col]=n.array(self.data[col])
        # only return data corresponding to steps beyond minstep
        minsample = -1
        datalen = len(self.steps)
        for i in range(0,datalen):
            if self.steps[i] >= minstep:
                minsample = i
                break
        if minsample >= 0:
            for col in column:
                output[col] = data[col][minsample:datalen]
        return output
    def addPosline(self, line):
        """
            Add input line to the various pos-related data structures.

            Presumes poly3d data.
        """
        i = 0

        tokens = re.split('\s+',line,maxsplit=1)
        if tokens[0] in self.definitions[-1]:
            line = self.definitions[-1][tokens[0]] + ' ' + line.lstrip(tokens[0])+' '
        if tokens[0] in self.injavis_params:
            self.addInjavisParam(line)
        elif re.match('^def',line):
            self.addDef(line)
        elif re.match('^shape\s+',line):
            self.setShapePrefix(line)
        elif ( re.match("^boxMatrix",line) ):
            lst = line.rstrip().split()
            self.boxMatrix_list.append([float(i) for i in lst[1:len(lst)]])
        elif ( re.match("^box",line) ):
            lst = line.rstrip().split()
            self.boxMatrix_list.append([0 for i in range(9)])
            for i in range(3):
              self.boxMatrix_list[-1][4*i] = float(lst[i+1])

        elif ( re.match("^eof",line) ):
            self.definitions.append(dict())
            self.shape_prefix = ''
            self.particle_list.append([])
            self.position_list.append([])
            self.quaternion_list.append([])
            self.object_list.append([])
        else:
            self.addParticle(self.shape_prefix + line)

    def getPos(self, minstep=0):
        """
            Return dictionary of positions, quaternions, and box matrix.

            To do: optionally return specific steps / frames
        """
        ret = {}
        ret['positions'] = self.position_list
        ret['quaternions'] = self.quaternion_list
        ret['boxMatrix'] = self.boxMatrix_list
        ret['particles'] = self.particle_list
        ret['objects'] = self.object_list
        ret['defs'] = self.definitions
        ret['frames'] = self.frame_list
        ret['injavis_params']=self.injavis_params
        return ret

    def addDef(self,line):
        tokens = re.split('\s+',line,maxsplit=2)
        key = tokens[1]
        val = tokens[2].strip('"')
        self.definitions[-1][key] = val
        if not self.shape_prefix=='':
            raise RuntimeError('Cannot use \'shape\' and \'def\' macros in the same pos frame')

    def setShapePrefix(self,line):
        tokens = re.split('\s+',line,maxsplit=1)
        shape = tokens[1].strip('"')
        self.shape_prefix = shape+' '
        if len(self.definitions[-1])>0:
            raise RuntimeError('Cannot use \'shape\' and \'def\' macros in the same pos frame')

    def addInjavisParam(self,line):
        tokens = re.split('\s+',line,maxsplit=1)
        var = tokens[0]
        val = tokens[1].strip('\"')
        self.injavis_params[var]=val

    def addParticle(self,line):
        tokens = re.split('\s+',line.strip())
        shape = tokens[0]

        #spheres are only shape without a quaternion; an empty array is append to quaternion_list to keep
        #all the lists aligned

        # Can get speed up through optimization here...
        # Don't need to make a new object every time

        if   shape == 'sphere':
            self.particle_list[-1].append(Sphere(float(tokens[1]),tokens[2]))
            self.position_list[-1].append(n.array([float(p) for p in tokens[3:6]]))
            self.quaternion_list[-1].append(n.array([]))

        elif shape == 'jsphere':
            self.particle_list[-1].append(JanusSphere(float(tokens[1]),float(tokens[2]),tokens[3:5]))
            self.position_list[-1].append(n.array([float(p) for p in tokens[5:8]]))
            self.quaternion_list[-1].append(n.array([float(q) for q in tokens[8:12]]))

        elif shape == 'poly3d':
            N = int(tokens[1])
            verts=n.array([float(p) for p in tokens[2:2+3*N]]).reshape(N,3)
            self.particle_list[-1].append(Poly3D(verts,tokens[3*N+2]))
            self.position_list[-1].append(n.array([float(p) for p in tokens[3*(N+1):3*(N+2)]]))
            self.quaternion_list[-1].append(n.array([float(p) for p in tokens[3*(N+2)::]]))

        elif shape == 'poly':
            N = int(tokens[1])
            verts=n.array([float(p) for p in tokens[2:2+3*N]]).reshape(N,3)
            self.object_list[-1].append(Poly(verts,tokens[3*N+2]))

        elif shape == 'bond':
            self.object_list[-1].append(Bond(float(tokens[1]),tokens[2],n.array([float(p) for p in tokens[3::]])))

        elif shape == 'ellipsoid':
            self.particle_list[-1].append(Ellipsoid(n.array([float(p) for p in tokens[1:4]]),tokens[4]))
            self.position_list[-1].append(n.array([float(p) for p in tokens[5:8]]))
            self.quaternion_list[-1].append(n.array([float(p) for p in tokens[8::]]))

        elif shape == 'jellipsoid':
            self.particle_list[-1].append(JanusEllipsoid(n.array([float(p) for p in tokens[1:4]]),
                                    n.array([float(p) for p in tokens[4:7]]),
                                    float(tokens[7]),
                                    tokens[8:10]))
            self.position_list[-1].append(n.array([float(p) for p in tokens[10:13]]))
            self.quaternion_list[-1].append(n.array([float(p) for p in tokens[13::]]))

        elif shape == 'polySphere':
            diam = float(tokens[1])
            N = int(tokens[2])
            verts = n.array([float(p) for p in tokens[3:3*(n.abs(N)+1)]]).reshape(N,3)
            self.particle_list[-1].append(PolySphere(diam,N,verts,tokens[3*(n.abs(N)+1)]))
            self.position_list[-1].append(n.array([float(p) for p in tokens[3*(n.abs(N)+1)+1:3*(n.abs(N)+2)+1]]))
            self.quaternion_list[-1].append(n.array([float(p) for p in tokens[3*(n.abs(N)+2)+1::]]))

class Shape(object):
    def __init__(self):
        self.hash=0
    def writeDef(self):
        return self.definition
    def __eq__(self,other):
        return self.definition==other.definition
    def __str__(self):
        return self.definition
    def __hash__(self):
        return self.hash
    def calc_hash(self):
        self.hash = int(n.prod([n.abs(ord(c))+1 for c in list(self.definition)])+n.sum([ord(c) for c in list(self.definition)]))

class Sphere(Shape):
    def __init__(self,size,color):
        self.size = size
        self.definition ='"sphere %5.8f %s "'%(size, color)
        self.calc_hash()
class Poly(Shape):
    def __init__(self,vert,color):
        super(self.__class__, self).__init__()
        self.vert=vert
        self.definition= "poly "+str(len(vert))+" "
        for v in vert:
            self.definition+=" " +"%f %f %f "%(v[0],v[1],v[2])
        self.definition += color
        self.calc_hash()

class Poly3D(Shape):
    def __init__(self,vert,color):
        super(self.__class__, self).__init__()
        self.vert=vert
        self.color=color
        self.definition= "\"poly3d %d "%(len(vert))
        for v in vert:
            self.definition+=" " +"%f %f %f "%(v[0],v[1],v[2])
        self.definition += color+"\""
        self.calc_hash()
class JanusSphere(Shape):
    def __init__(self,size,balance,colors,com=None,quat=None):
        super(self.__class__,self).__init__()
        self.balance=balance
        self.size=size
        print(colors)
        self.definition = '"jsphere %f %f %s %s"'%(self.size,self.balance,colors[0],colors[1])
        self.calc_hash()
class Ellipsoid(Shape):
    def __init__(self,shape,color):
        super(self.__class__,self).__init__()
        self.shape=shape
        self.definition = '"ellipsoid %5.8f %5.8f %5.8f %s "'%(self.shape[0],
                            self.shape[1],self.shape[2],color)
        self.calc_hash()

class JanusEllipsoid(Shape):
    def __init__(self,shape,j_norm,balance,colors,com=None,quat=None):
        super(self.__class__,self).__init__()
        self.balance=balance
        self.shape=shape
        self.j_norm=j_norm
        self.definition = '"jellipsoid %5.8f %5.8f %5.8f %5.8f %5.8f %5.8f %5.8f %s %s "'%(self.shape[0],
                            self.shape[1],self.shape[2],self.j_norm[0],self.j_norm[1],self.j_norm[2],self.balance,
                            colors[0],colors[1] )
        self.calc_hash()

class PolySphere(Shape):
    def __init__(self,size,N,verts,color):
        super(self.__class__,self).__init__()
        self.size = size
        self.N = N
        self.verts = verts
        self.definition = '"polySphere %5.8f %d '%(self.size,len(verts))
        for v in verts:
          self.definition += ' %5.8f %5.8f %5.8f '%(v[0],v[1],v[2])
        self.definition += ' %s "'%(color)
        self.calc_hash()

class Bond(Shape):
    def __init__(self, diameter, color, verts):
        super(self.__class__,self).__init__()
        self.diameter = diameter
        self.verts = verts
        self.definition = 'bond %5.8f %s %5.8f  %5.8f %5.8f %5.8f %5.8f %5.8f '%(self.diameter,color,verts[0],verts[1],verts[2],
                                                                          verts[3],verts[4],verts[5])
        self.calc_hash()
        self.color=color

class Simulation:
    """
        A class to simplify interacting with incsim processes from python.

        An executable (incsim) simulator must be specified to the constructor.
        Other arguments are optional.

        There is no particularly good way to test for the availabilty
        of output at this time. Future implementations of this class
        may use threads and queues to solve this issue.
    """
    def __init__(self, simulator=None, input=None, output=None, initialize=None, seed=None, append=False, unbuffered=False):
        self.ifile = None
        if isinstance(output, file):
            self.output = output.name
        else:
            self.output= output
        self.ofile = None
        self.proc = None
        self.mc = None
        self.initialized = False
        self.instructions = list()
        if seed is None:
            self.seed = str(os.getpid())
        else:
            self.seed = str(seed)
        if not output is None: self.setOutput(output, append)
        if isinstance(simulator, str):
            if os.access(simulator, os.X_OK):
                self.mc = simulator
        if self.mc is None:
            raise RuntimeError('No valid simulator specified')
            return None
        if not input is None: self.setInput(input)
        if not simulator is None and not initialize is None:
            self.initialize(initialize, unbuffered)
    # need to properly kill the simulation when it is done...
    def __del__(self):
        """
            This destructor only kills the simulator to avoid orphan processes.
            To exit cleanly, use the .quit() method.
        """
        # to do: should be prepared to catch IOError: Broken pipe
        if self.proc.poll() is None: self.proc.terminate()
        self.proc.wait()
    def setInput(self, input=None):
        """
            Set the name of the input pos file. Run automatically
            if filename is specified to the constructor.
        """
        self.ifile = input
    def setOutput(self, output=None, append=False):
        """
            Set the output file to be used during the simulation
            Called automatically if output filename is given during
            creation of the Simulation object. Must be done before
            initialize is called or stdout will be used.

            output may be a filename or an open filehandle. For example,
            to use STDOUT, import sys and use setOutput(sys.stdout).

            If you need to read output directly from the simulation process,
            specify a filename of 'pipe' and refer to the subprocess module
            for more on reading from the proc object. Be very careful if you
            try this because it is easy to get deadlocked if your program
            blocks while waiting for output from the simulator. It is better
            to send the output to a file and use the File class to read from
            that file. Don't forget to flush the input/output streams before
            reading.
        """
        if output is None and not self.ofile is None:
            output = self.ofile
        if type(output) is file:
            self.ofile = output
        elif re.search('^pipe$', str(output), re.I):
            self.ofile = subprocess.PIPE
        else:
            if append == True:
                self.ofile = open(output, 'a')
            else:
                self.ofile = open(output, 'w')
        #return self.ofile
    def initialize(self, initialize=None, unbuffered=False):
        """
            Open a process handle to a running simulation. Optionally
            initialize the simulator with some initial instructions,
            such as to load a pos file and set the random number seed.

            If unbuffered is True, the flush command will be unnecessary,
            but I/O may be slower and the simulation may hang if output is
            not dealt with in a timely fashion as when using pipes.
        """
        if unbuffered:
            bufsize=0
        else:
            bufsize=-1
        if not isinstance(self.ofile, file) and self.ofile != subprocess.PIPE:
            raise RuntimeError('No output filehandle available')
            return
        self.proc = subprocess.Popen(self.mc, bufsize=bufsize, executable=self.mc, stdin=subprocess.PIPE, stdout=self.ofile, stderr=subprocess.PIPE, shell=False, universal_newlines=True)
        self.initialized = True
        if initialize is None and not self.ifile is None:
            initialize = "load %s seed %s" % (self.ifile, self.seed)
        if not initialize is None: self.send(initialize)
        return self
    def flush(self):
        """
            A convenience method to flush stdout and stdin of the running
            simulation. Note that any output that results from instructions
            activated by a flush will not necessarily be available right away.
        """
        if isinstance(self.proc.stdout, file): self.proc.stdout.flush()
        if isinstance(self.proc.stdin, file): self.proc.stdin.flush()
    def send(self, instructions=None):
        """
            Send an instruction or list of instructions to the simulator.
            Note that by default the instructions are buffered. If you need to
            make sure instructions are sent at a certain point in your script,
            use instance.proc.stdin.flush() or instance.flush() to flush the buffer.

            Use incsim.File and the getData method to get intermediate results.
            See the unpack method as an example.
        """
        if not self.initialized: self.initialize()
        if self.proc is None:
            raise RuntimeError('no simulation running')
            return
        # repeat previous instruction(s) if none is given
        if instructions is None and len(self.instructions) > 0:
            instructions = self.instructions
        if not isinstance(instructions, list):
            instructions = list([instructions])
        self.instructions = list()
        for line in instructions:
            self.proc.stdin.write(line)
            if not re.search('\n$', line): self.proc.stdin.write('\n')
            self.instructions.append(line)
    # def unpack(self, scale=0.99):
    #     """
    #         Run a few steps in NVT to set
    #         packing fraction to 99% of it's current value.
    #     """
    #     if not self.initialized: self.initialize()
    #     if self.proc is None:
    #         raise RuntimeError('no simulation running')
    #         return
    #     if not isinstance(self.ofile, file):
    #         raise RuntimeError("Don't know where to get results for unpack")
    #         return
    #     filename = self.ofile.name
    #     self.ofile.flush()
    #     output = open(filename, 'rU')
    #     output.seek(0,2)
    #     oldposition = output.tell()
    #     oldsize = os.path.getsize(filename)
    #     newsize = oldsize
    #     self.send("steps 1 skip 1 NVT next")
    #     self.flush()
    #     # need to wait for output to appear...
    #     # this is still a bit of a race condition and should be handled with
    #     # queue objects from the threading library
    #     while newsize == oldsize:
    #         self.ofile.flush()
    #         newsize = os.path.getsize(filename)
    #         time.sleep(1)
    #     output.seek(oldposition, 0)
    #     datfile = File(output)
    #     output.close()
    #     data = datfile.getData("Packing")
    #     pf = data['Packing'][len(data['Packing']) - 1 ]
    #     pf *= scale
    #     self.send("setpf %.3f next" % pf)
    def runfor(self, n=1, instructions=None):
        """
            Use as a simply looping construct to run instructions n times.
            If no instructions are given, issue 'next'. instructions may be a list.
        """
        n = int(n)
        if not self.initialized: self.initialize()
        if self.proc is None:
            raise RuntimeError('no simulation running')
            return
        if instructions is None: instructions='next'
        for a in range(0, n):
            self.send(instructions)
    def quit(self):
        """
            Send the quit message to incsim, if it hasn't been sent already.
            Wait for the simulator to end, and return text of STDERR.
            Note that if the process has already been terminated, this method
            will result in IOError: Broken pipe, so you may want to wrap this
            call in a try...except clause.
        """
        # to do: should be prepared to catch IOError: Broken pipe
        m = re.search('quit[\n\s]*$', self.instructions[ len(self.instructions) - 1 ] )
        if not m:
            self.send('quit')
        self.proc.stdin.flush()
        result = self.proc.communicate()
        self.proc.wait()
        self.initialized=False
        return result[1]
