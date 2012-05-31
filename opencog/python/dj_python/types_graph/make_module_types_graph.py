import fileinput                         
import networkx as nx
from opencog.atomspace import types as t
G = nx.DiGraph()
for line in fileinput.input(inplace = False):
    num = fileinput.lineno()           
    if line.find('=') != -1:
	line = line.replace('\n','')
	t = line.split('(')
	line = t[1]
	a = t[0].split(' ')[0]                                  # opencog::CHILD_NODE
	child = a.split(':')[2].strip(' ')           # CHILD_NODE
	tt = line.split(',')
	ttt = tt[0].strip(' ')                               # opencog::ATOM
	ta = ttt.split(':')
	parent = ta[2]                                # ATOM
	tttt = tt[1].split(')')
	child_name  = tttt[0].strip(' "')                           # Atom 
	# parent -> child
	G.add_edge(parent,child)
	G.node[child]['type_name']=child_name
#AtomTypes_Graph = nx.DiGraph() 
print "import networkx as nx" 
print "types_graph = nx.DiGraph()" 
for node in G.nodes():
    for child in G.neighbors(node):
	parent = G.node[node]['type_name']
	child = G.node[child]['type_name']
        #AtomTypes_Graph.add_edge(parent,child)
	#template =  "M.node[%s]['type'] =  t.%s"
	#parent_code = template %(parent,parent)
	#child_code = template %(child,child)
	print "types_graph.add_edge('%s','%s')" %(parent,child)
	#AtomTypes_Graph.add_edge(parent,child)
print "types_graph.remove_edge('Atom','Atom')" 
#AtomTypes_Graph.remove_edge('Atom','Atom')
	
#print "the total num of line is:%d" % num
