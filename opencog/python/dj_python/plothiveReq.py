import time
import opencog.cogserver
from tree import FakeAtom,fake_from_real_Atom
import util
from opencog.atomspace import types,AtomSpace,Handle,Atom

import rpy2.robjects as robjects
#from dataAnalysisR import DataAnalysisR
import opencog.atomspace

class PlotHiveReq(opencog.cogserver.Request):
    def run(self,args,atomspace):
	    print "PlotHiveRequest is running ....."
	    #import R packages
	    robjects.r('''
		    library(HiveR)
		    library(grid)
		    library(RColorBrewer)
		    source("myplotHive.R")
		    ''')
	    p = ASpace_DataAnalysisR(atomspace,args[0])
	    p.extractData()
	    p.plotHive()
	    time.sleep(1)
class AxisArg(object):
	"""an strucure that store the information of axis extract from arguments"""
	def __init__(self, name, color = "green",node_color = "yellow" ,node_size = 1):
	    self.name = name
	    self.color = color
	    self.node_color = node_color
	    self.node_size = float(node_size)
	    self.node_type = name
class EdgeArg(object):
	"""an strucure that store the information of links  extract from arguments"""
	def __init__(self, type ,color = "blue" ,weight = 1):
	    self.type = type
	    self.color = color
	    self.weight = float(weight)
class NodesR(object):
	"""strucure to write to R ,as nodes on axis to display"""
	def __init__(self, id = [],lab = [],axis = [],radius = [],size = [],color = []):
	    self.id = id
	    self.lab = lab
	    self.axis = axis
	    self.radius = radius
	    self.size = size
	    self.color = color
class EdgesR(object):
	"""strucure to write to R ,as edges to display"""
	def __init__(self, id1 = [], id2 = [], weight = [], color = []):
	    self.id1 = id1
	    self.id2 = id2
	    self.weight = weight
	    self.color = color
class GraphicR(object):
    """docstring for GraphicR"""
    def __init__(self,axisArgs, edgesArg):
	self.nodesR = NodesR()
	self.edgesR = EdgesR()
	#data help to determine if the type of an edge or a node is user required 
	self.axisArgs = axisArgs
	self.axisReal = []
	self.edgesReal = []
	self.edgesArg = edgesArg
	self.numNodesAx = []
	self.edgesPsb = []
	#default color of edges
	self.defCols = ["yellow","blue","red","green"]
	self.colId = 0
	self.DEFAULT_EDGE_COLOR = "white" 
	self.DEFAULT_EDGE_WEIGHT = 1            



    def addNode(self,node):
	"""transfer from atom_node to nodesR"""
	assert len(self.axisArgs) > 0
	try:
	    self.nodesR.id.index(node._handle_value)
	    return
	except Exception, e:
	    noAx = 0
	    no = 0
	    for ax in self.axisArgs:
		if node.type_name == ax.name:
		    try:
			noAx = self.axisReal.index(ax) 
		    except Exception,e:
			self.axisReal.append(ax)
			self.numNodesAx.append(0)
			noAx = len(self.numNodesAx) - 1
		    argAx = ax
		    self.numNodesAx[noAx] += 1
		    no = self.numNodesAx[noAx]
		    break
	    else:
		assert False
	    size = float(argAx.node_size)
	    self.nodesR.id.append(node._handle_value)
	    if node.name == "" :
	    #a link
		node.name =  node.type_name
	    self.nodesR.lab.append(node.name)
	    self.nodesR.axis.append(noAx + 1)
	    self.nodesR.radius.append(no*size*2+size)
	    self.nodesR.size.append(size)
	    self.nodesR.color.append(argAx.node_color)
    def addNodes(self,nodes):
	for node in nodes:
	    self.addNode(node)

    def validEdge(self,edge,nodes):
	"""make sure the type links and it targets are required type"""
	assert len(self.axisArgs) > 0
	if len(self.edgesArg) > 0:
	    for arg in self.edgesArg:
		if edge.type_name == arg.type:
		    try:
		        self.edgesReal.index(arg) 
		    except Exception,e:
			self.edgesReal.append(arg)

		    break
	    else:
		#print "invalid link:" + edge.type_name + "****************" 
		return False
	#determine if the outs of link is required node type
	for node in nodes:
	    for ax in self.axisArgs:
		if node.type_name == ax.node_type:
		    break
	    else:
		#print "invalid link:" + edge.type_name + "with nodes:****************" 
		#for node in nodes:
		    #print node.type_name
		return False
	#
	#print "valid link:"  + edge.type_name
	return True
    def addEdge(self,edge,nodes):
	try:
	    i = 0
	    for node in nodes:
	    #
		i += 1
		self.edgesR.id2.append(nodes[i]._handle_value)
		self.edgesR.id1.append(node._handle_value)

		for arg in self.edgesArg:
		#arguments: nodes + edges
		    if edge.type_name == arg.type:
			self.edgesR.color.append(arg.color)
			self.edgesR.weight.append(arg.weight)
			break
		else:
		#arguments: nodes
		    for arg in self.edgesPsb:
			if edge.type_name == arg.type:
			    self.edgesR.color.append(arg.color)
			    self.edgesR.weight.append(arg.weight)
			    break
		    else:
		    #first time
			if len(self.defCols) > self.colId:
			    color = self.defCols[self.colId]
			    self.colId += 1
			else:
			    color = self.DEFAULT_EDGE_COLOR
			edge = EdgeArg(edge.type_name,color,self.DEFAULT_EDGE_WEIGHT)
			self.edgesPsb.append(edge)
			self.edgesR.color.append(edge.color)
			self.edgesR.weight.append(edge.weight)
	except Exception, e:
	    #todo KeyError is right
	    #else raise
	    pass

	
class DataAnalysisR(object):
    """class which extract data from 'source' and display it ,as user required""" 
    def __init__(self,source,arguments):
	self.typesMap = {
		    'Node':types.Node,
		    'ConceptNode':types.ConceptNode,
		    'Link' :types.Link,
		    'InheritanceLink' :types.InheritanceLink,
		    'GroundedSchemaNode': types.GroundedSchemaNode,
		    'ListLink':types.ListLink,
		    'EvaluationLink':types.EvaluationLink
		   }
	self.arguments = arguments
	self.source = source
	#data extract from user input
	self.axisArgs = []
	self.edgesArg = []
	self.numNodesAx = []
	#determine if print behavior
	self.printLable = False
	self.autoSize = False
	#init data
	self.parseArgs(self.arguments)
	self.graphicR = GraphicR(self.axisArgs,self.edgesArg)
    #def addNode(self,node):
        #"""docstring for addNode"""
	#self.graphicR.fillNode(node)
	#if self.printLable:
	    #try:
		#f = open("nodelabels.csv" ,'a')
		#noNode = 0
		#for label in self.graphicR.nodesR.label:
		    #line = "%s,%s,90,20,0,0.2,-0.5\n" % (label,label)
		    #f.writelines(line)
		    #noNode += 1
	    #except Exception, e:
		#print "Failed to print atom names!"
		#self.printLable = False
	    #finally:
		#f.close()

    def _getLinks(self,type):
        """docstring for getLinks"""
	pass
    def _nodesFromLink(self,link):
        pass

    ##
    # @brief :extract data from atomspace as user required
    #
    # @param arg :a string like "ConceptNode-color-node_color-node_size:...:Link-color-node_color-node_size|Link-color-weight|shortby" 
    #
    # @return :none
    def extractData(self):
	"""extract data from atomspace as user required""" 
	try:
	    if len(self.edgesArg) > 0 and len(self.axisArgs) > 0:
	    #arguments: nodes + edges
		for edge in self.edgesArg:
		    #links = atomspace.get_atoms_by_type(self.typesMap[edge.name])
		    links = self._getLinks(self.typesMap[edge.type])
		    for link in links:
			nodes = self._nodesFromLink(link)
			if len(nodes) > 0:   #only non empty links
			    if self.graphicR.validEdge(link,nodes):
				self.graphicR.addEdge(link,nodes)
				self.graphicR.addNodes(nodes)

	    elif len(self.axisArgs) > 0:
	    #only with node args and all possible links ,len(self.edgesArg)=0
		    links = self._getLinks(types.Link)
		    for link in links:
			#fill nonEmpty link
			nodes = self._nodesFromLink(link)
			if len(nodes) > 0:
			    if self.graphicR.validEdge(link,nodes):
				self.graphicR.addEdge(link,nodes)
				self.graphicR.addNodes(nodes)

	    else:
		print "Sorry, there is not information required!" 

	except Exception, e:
	    print e

    def plotHive(self):
	"""plot the data"""
	##?
	self._dataPyToR()
	robjects.r('''
	#print(data)
	    pdf(file = "t.pdf" )
	    myplotHive(data,axLabs =  axNames, axLab_pos =  c(rep(10,numAxs)), axLab_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2)
			   ,dr_nodes =FALSE,anNode_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2),edgesDsp.type = edgeDsp.type, edgesDsp.color = edgeDsp.color)
	    dev.off()
	         ''')
	
        #cmd = '''
	#print(data)
	    #dspNode = FALSE

	    #pdf(file = "t.pdf" )
		  #myplotHive(data)
		  #plotHive(data)
	    ##myplotHive(data,axLabs =  axNames, axLab_pos =  c(rep(10,numAxs)), axLab_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2)
			   ##,%s ,anNode_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2),edgesDsp.type = edgeDsp.type, edgesDsp.color = edgeDsp.color)
	    #dev.off()

	    ##jpeg(file = "t.jpeg" ,width = 800,height = 400)
	    ###myplotHive(data,dr_nodes = dspNode,axLabs =  axNames, axLab_pos =  c(rep(10,numAxs)), axLab_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2)
			   ###,%s ,anNode_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2))

	    ##myplotHive(data,axLabs =  axNames, axLab_pos =  c(rep(10,numAxs)), axLab_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2)
			   ##,%s ,anNode_gpar =  gpar(col =  "yellow", fontsize =  14, lwd =  2),edgesDsp.type = edgeDsp.type, edgesDsp.color = edgeDsp.color)
	    ##dev.off()
	    ##rot = c(rep(90,numAxs)),
		 #'''
	#if self.printLable:
	    #tmp = 'anNodes = "nodelabels.csv"' 
	    #cmd = cmd % (tmp,tmp)
	#else:
	    #cmd = cmd % ('','')
	#robjects.r(cmd)
	
    def _dataPyToR(self):
	"""fill the HivePlotData data strucure of R, before plot it"""
	axNames = []
	axCols = []
	for ax in self.graphicR.axisReal:
	    axNames.append(ax.name)
	    axCols.append(ax.color)
	tc = []
	tt = []
	if len(self.graphicR.edgesReal) > 0:
	    for edge in self.graphicR.edgesReal:
		tt.append(edge.type)
		tc.append(edge.color)
		print(edge.type)
	if len(self.graphicR.edgesPsb) > 0:
	    for edge in self.graphicR.edgesPsb:
		tt.append(edge.type)
		tc.append(edge.color)
		print(edge.type)
	robjects.globalenv['edgeDsp.color'] = robjects.StrVector(tc)
	robjects.globalenv['edgeDsp.type'] = robjects.StrVector(tt)

	robjects.globalenv['axNames'] = robjects.StrVector(axNames)
	robjects.globalenv['axCols'] = robjects.StrVector(axCols)
	robjects.globalenv['idV'] = robjects.IntVector(self.graphicR.nodesR.id)
	robjects.globalenv['labV'] = robjects.StrVector(self.graphicR.nodesR.lab)
	robjects.globalenv['axisV'] = robjects.IntVector(self.graphicR.nodesR.axis)
	robjects.globalenv['radiusV'] = robjects.FloatVector(self.graphicR.nodesR.radius)
	robjects.globalenv['sizeV'] = robjects.FloatVector(self.graphicR.nodesR.size)
	robjects.globalenv['colorV'] = robjects.StrVector(self.graphicR.nodesR.color)
	robjects.globalenv['id1V'] = robjects.IntVector(self.graphicR.edgesR.id1)
	robjects.globalenv['id2V'] = robjects.IntVector(self.graphicR.edgesR.id2)
	robjects.globalenv['weightV'] = robjects.FloatVector(self.graphicR.edgesR.weight)
	robjects.globalenv['e_colorV'] = robjects.StrVector(self.graphicR.edgesR.color)
	numAxs = len(self.graphicR.numNodesAx)
	robjects.globalenv['numAxs'] = numAxs
	numNodes = len(self.graphicR.nodesR.id)
	numEdges = len(self.graphicR.edgesR.id1)
	robjects.r('''
		  data <- ranHiveData(nx = numAxs,type = "2D" )
		  nodes<-data.frame(id = idV, lab = labV, axis = axisV, radius = radiusV, size = sizeV, color = colorV)
		  edges<-data.frame(id1=id1V,id2=id2V,weight=weightV,color=e_colorV)
		  unclass(nodes$id)
		  unclass(nodes$lab)
		  unclass(nodes$axis)
		  unclass(nodes$color)
		  unclass(nodes$radius)
		  attr(nodes$id,"class")<-"integer"
		  attr(nodes$axis,"class")<-"integer"
		  attr(nodes$color,"class")<-"character"
		  nodes$lab<- labV
		  nodes$axis <- axisV
		  nodes$radius<-radiusV
		  nodes$color<-colorV
		  data$nodes<-nodes

		  unclass(edges$id1)
		  unclass(edges$id2)
		  unclass(edges$color)
		  attr(edges$id1,"class")<-"integer" 
		  attr(edges$id2,"class")<-"integer" 
		  attr(edges$color,"class")<-"character"
		  edges$color<-e_colorV
		  data$edges<-edges
		  data$axis.cols<-axCols


		 ''')
	cmd = 'data$desc <- " %s axis---------%s nodes-------%s edges" ' % (numAxs,numNodes,numEdges)
	robjects.r(cmd)
	#robjects.r('''
		  #print(data)
		 #''')
	
	
	
	

    def parseArgs(self,args):
        """parse arguments user input"""
	try:
	#extract arguments
	    errFmt = Exception('wrong arguments format!input string arg like "ConceptNode-color-node_color-node_size:...:|Link-color-size:...:|order:printlabel:self.autoSize"')
	    ###########arguments
	    cmdList = args.split('|')
	    if len(cmdList) != 3:
		raise errFmt

	    dspAxis = cmdList[0] # string like "ConceptNode-color-node_color-node_size:...:Link-color-node_color-node_size" 
	    dspEdges = cmdList[1]
	    options = cmdList[2]
	    ###########fill options
	    args = options.split(':')
	    if len(args) == 3 :
		order = args[0]
		if args[1].lower() == "false":
		    self.printLable = False
		elif args[1].lower() == "true":
		    self.printLable = True
		else:
		    raise errFmt

		if args[2].lower() == "false":
		    self.autoSize = False
		elif args[2].lower() == "true":
		    self.autoSize = True
		else:
		    raise errFmt
	    else:
		raise errFmt
	    if self.printLable:
		try:
		    header = "node.lab,node.text,angle,radius,offset,hjust,vjust\n" 
		    f = open("nodelabels.csv" ,'w')
		    f.writelines(header)
		except Exception, e:
		    print "Failed to print atom names!" 
		    self.printLable = False
		finally:
		    f.close()
	    ###########fill self.axisArgs -- list of type AxisArg
	    if len(dspAxis) > 0:
		args = dspAxis.split(':') #list of strings like "ConceptNode-color-node_color-node_size" 
		for arg in args:
		    self.numNodesAx.append(0)
		    data = arg.split('-')
		    if len(data) == 4:
			ax = AxisArg(data[0],data[1],data[2],data[3])
		    else:
			raise errFmt
		    self.axisArgs.append(ax)
	    ###########fill self.edgesArg -- list of type EdgeArg
	    if len(dspEdges) > 0:
		args = dspEdges.split(':')
		for arg in args:
		    data = arg.split('-')
		    if len(data) == 3:
			edge = EdgeArg(data[0],data[1],data[2])
		    else:
			raise errFmt
		    self.edgesArg.append(edge)
	#todo exit
	except KeyError,e:
	    print errFmt
	    print e
	except IndexError,e:
	    print errFmt
	    print e
	except Exception, e:
	    print errFmt
	    print e
class ASpace_DataAnalysisR(DataAnalysisR):
    """docstring for Aspace_DataAnalysis"""
    def __init__(self, source,args):
	super(ASpace_DataAnalysisR, self).__init__(source,args)
	self.atomspace = source

    def _getLinks(self,type):
        """docstring for __getLinks"""
	return  self.atomspace.get_atoms_by_type(type)

    def _nodesFromLink(self,link):
	atoms = link.out
	fakeAtoms = []
	for atom in atoms:
	    fakeAtoms.append(fake_from_real_Atom(atom))
	return fakeAtoms
	

