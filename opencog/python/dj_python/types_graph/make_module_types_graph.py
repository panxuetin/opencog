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
print "from opencog.atomspace import types as t" 
print "types_graph = nx.DiGraph()" 
print "name_to_type = { }"
print "type_to_name = { }"
map_set = set()
map_set2 = set()
for node in G.nodes():
    for child in G.neighbors(node):
        parent = G.node[node]['type_name']
        child = G.node[child]['type_name']
        map_set.add('name_to_type["%s"] = t.%s'%(child,child))
        map_set2.add('type_to_name[t.%s] = "%s"'%(child,child))
        print "types_graph.add_edge('%s','%s')" %(parent,child)
print "types_graph.remove_edge('Atom','Atom')" 
for item in map_set:
    print item
for item in map_set2:
    print item
#AtomTypes_Graph.remove_edge('Atom','Atom')
	
#print "the total num of line is:%d" % num
