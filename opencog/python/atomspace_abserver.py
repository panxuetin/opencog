from viz_graph import Graph_Abserver
import networkx as ax
from types_inheritance import types_graph, name_to_type, is_a
#from pprint import pprint
from collections import defaultdict
from m_util import log
from m_adaptors import FakeAtom
log.setLevel(log.DEBUG)
log.use_stdout(True)

    
#from opencog.atomspace import Atom, types
class Atomspace_Abserver(Graph_Abserver):
    """ attention: not including isolate concpet node and empty link"""
    def __init__(self, a, e_types = ["Link"], n_types = ["Node", "Link"], inheritance = True):
        super(Atomspace_Abserver, self).__init__(a, e_types, n_types, inheritance)
        #self.valid_node_types += self.valid_edge_types
        self.no_type = { }

    def graph_info(self):
        '''docstring for graph_info''' 
        self.e_types = ["Link"]
        self.n_types = ["Node", "Link"]
        edges_info = defaultdict(int)
        nodes_info = defaultdict(set)

        for e_type in self.valid_edge_types:
            links = self._get_edges(e_type)
            for link in links:
                nodes = self._nodes_from_edge(link)
                if len(nodes) > 0:
                    if self.valid_edge(link,nodes):
                        #edges_info[link.type_name].add(link.name)
                        edges_info[link.type_name] += 1
                        for i, node in enumerate(nodes):
                            if is_a(node.type_name, "Node"):
                                nodes_info[node.type_name].add(node.name)
        log.debug("*******************************edges:" )
        for type_name, num in edges_info.iteritems():
            log.debug( type_name + ":  " + str(num))
            #pprint(edges)
        log.debug("*******************************nodes:" )
        for type_name, nodes in nodes_info.iteritems():
            log.debug(type_name + ": " + str(len(nodes)))
            #pprint(nodes)
    
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
