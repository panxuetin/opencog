from viz_graph import Viz_Graph, FakeAtom
import networkx as ax
from types_inheritance import types_graph, name_to_type, is_a
class Graph_Abserver(object):
    """docstring for Data_Abserver"""
    def __init__(self, source, e_types, n_types, inheritance = True):
        self.source = source
        self.valid_edge_types = e_types
        self.valid_node_types = n_types
        # types actually exist
        self.touched_edge_types = []
        self.touched_node_types = []
        self.graph = Viz_Graph()
        self.inheritance = inheritance


    def _nodes_from_edge(self,edge):
        pass

    def _get_edges(self,e_type):
        '''type of e_type is consistency with valid_edge_types '''
        pass
    def _edge_type(self,edge):
        '''type of edge is consistency with valid_edge_types '''
        pass

    def _node_type(self,node):
        '''type node is consistency with valid_node_types '''
        pass
    
    def _edge_is_a(self, source, target):
        '''type of source and target is consistency with valid_edge_types '''
        return source == target

    def _node_is_a(self, source, target):
        '''type of source and target is consistency with valid_node_types '''
        return source == target

    def write_dot(self, filename):
        '''docstring for write_dot''' 
        self.graph.write_dot(filename)

    def add_valid_edges(self):
        '''docstring for run()''' 
        # add edges of valid type
        # iterate over valid edges
        for e_type in self.valid_edge_types:
            edges = self._get_edges(e_type)
            for edge in edges:
                nodes = self._nodes_from_edge(edge)
                # none empty edges!
                if len(nodes) > 0:
                    if self.valid_edge(edge,nodes):
                        self.graph.add_edge(nodes[0], nodes[1])
                        # add edge attribute

    def valid_edge(self,edge,nodes):
	"""make sure the type edge and it targets are required type,
       if one of the target is invalid, then the edge is invalid
    """
        assert len(self.valid_edge_types) and len(self.valid_node_types) > 0
        #import pdb
        #pdb.set_trace()
        for arg in self.valid_edge_types:
            if self._edge_is_a(self._edge_type(edge), arg):
                try:
                    self.touched_edge_types.index(arg) 
                except Exception:
                    self.touched_edge_types.append(arg)
                break
        else:
            # invalid edge
            return False
        #determine if the outs of link is required node type
        for node in nodes:
            for arg in self.valid_node_types:
                if self._node_is_a(self._node_type(node), arg):
                    try:
                        self.touched_node_types.index(arg) 
                    except Exception:
                        self.touched_node_types.append(arg)
                    break
            else:
                # invalid node
                return False

        return True


    
from opencog.atomspace import Atom, types
class Atomspace_Abserver(Graph_Abserver):
    """docstring for Aspace_DataAnalysis"""
    def __init__(self, a, e_types = ["Link"], n_types = ["Node", "Link"], inheritance = True):
        super(Atomspace_Abserver, self).__init__(a, e_types, n_types, inheritance)
        #self.valid_node_types += self.valid_edge_types
        #self.num_link = { }

    def _get_edges(self,type):
        """docstring for __getEdges"""
        return  self.source.get_atoms_by_type(name_to_type[type])

    def _nodes_from_edge(self,edge):
        return edge.out

    def _edge_type(self,edge):
        return edge.type_name

    def _node_type(self,node):
        return node.type_name
	
    def _edge_is_a(self, source, target):
        if self.inheritance:
            return is_a(source,target)
        else:
            return source == target

    def _node_is_a(self, source, target):
        return ax.has_path(types_graph, target, source)

    def add_valid_edges(self):
        '''docstring for run()''' 
        # add edges of valid type
        # iterate over valid edges
        for e_type in self.valid_edge_types:
            links = self._get_edges(e_type)
            for link in links:
                nodes = self._nodes_from_edge(link)
                # none empty edges!
                if len(nodes) > 0:
                    if self.valid_edge(link,nodes):
                        # make the linkname uniqueness
                        link_name = link.type_name + str(link.h.value())
                        for i, node in enumerate(nodes):
                            if is_a(node.type_name, "Link"):
                               node_name = node.type_name + str(node.h.value())
                            else:
                                node_name = node.name
                            #print "%s -> %s" %(link_name,node_name)
                            self.graph.add_edge(link_name,node_name)
                            # maintain order in the list
                            self.graph.set_edge_attr(link_name, node_name, order = str(i))

                            atom = FakeAtom(node.type, node_name, node.tv, node.av)
                            self.graph.set_node_attr(node_name, atom = atom)
                            #self.graph.set_node_attr(node_name, shape = "point")

                            atom = FakeAtom(link.type, link.name, link.tv, link.av)
                            self.graph.set_node_attr(link_name, atom = atom)
                            #self.graph.set_node_attr(link_name, shape = "point")
