##
# @file viz_graph.py
# @brief 
# @author Dingjie.Wang
# @version 1.0
# @date 2012-07-31

import networkx as nx
from collections import defaultdict


class Tree(object):
    """docstring for Tree"""
    def __init__(self, op, children = []):
        assert type(op) != type(None)
        self._op = op
        self._children = children

    def is_leaf(self):
        '''docstring for is_leaf''' 
        return True if self._children else False

    def get_op(self):
        return self._op

    def set_op(self, value):
        self._op = value
    op = property(get_op, set_op)

    def get_children(self):
        return self._children
    def set_children(self, value):
        self._children = value
    children = property(get_children, set_children)    

    def __str__(self):
        if self.is_leaf():
            return str(self.op)
        else:
            return '(' + str(self.op) + ' '+ ' '.join(map(str, self.children)) + ')'

    def __repr__(self):
        return str(self)

def trees_to_forest( trees ):
    '''docstring for trees_to_forest''' 
    assert type(trees) == list
    return Tree("forest", trees)

def tree_to_graphic(tree, graph):
    ''' transfer a simpler and more efficient tree StructureNode
        to Viz_Graph for visualisation purpose
    ''' 
    if tree.children:
        # inner node
        assert isinstance(tree.op, str)
        for i, child in enumerate(tree.children):
            # make name of tree node unique
            child_name = graph.unique_id(child.op)
            child.op = child_name
            child_name = tree_to_graphic(child, graph)
            graph.add_edge(tree.op, child_name, order = i)
        return tree.op
    else:
        # leaf node 
        return tree.op


        
import pygephi
class Gephi_Output:

    def __init__(self):
        self.gephi = pygephi.JSONClient('http://localhost:8080/workspace0', autoflush=True)
        self.gephi.clean()
        self.default_node_attr = {'size':10, 'r':0.0, 'g':0.0, 'b':1.0, 'x':1}
        self.default_edge_attr = { }

    #def start(self):
        #pass

    #def stop(self):
        #pass
    def write(self, filename = None):
        pass
    ## @todo interate label to attr
    def output_node(self, node_id, label = None, attr = None):
        if attr:
            self.gephi.add_node(str(node_id), label=label,  **attr)
        else:
            self.gephi.add_node(str(node_id), label=label,  **self.default_node_attr)

    def output_edge(self, source, target, edge_id, label = None, directed = True, attr = None):
        self.gephi.add_edge(str(edge_id), source, target, directed, label=label)

class Dotty_Output(object):
    """docstring for Dot_output"""
    def __init__(self):
        self.body = "" 

    def output_node(self, node_id, label = None, attr = None):
        '''docstring for output_node''' 
        line =  '"%s" '% str(node_id) 
        if attr:
            line += "[%s]" 
            str_attr = "" 
            try:
                str_attr += "color=%s," % attr['color']
            except Exception:
                pass
            try:
                str_attr += "shape=%s," % attr['shape']
            except Exception:
                pass
            try:
                str_attr += "style=%s," % attr['style']
            except Exception:
                pass
            str_attr = str_attr.strip(',')
            line = line % str_attr
        self.body += line + ";\n" 

    def output_edge(self, source, target, edge_id = None, label = None, directed = True, attr = None):
        line =  '"%s" -> "%s" ' %(str(source), str(target))
        if attr:
            line += "[%s]" 
            str_attr = "" 
            try:
                str_attr += "color=%s," % attr['color']
            except Exception:
                pass
            try:
                str_attr += "shape=%s," % attr['shape']
            except Exception:
                pass
            try:
                str_attr += "style=%s," % attr['style']
            except Exception:
                pass
            try:
                str_attr += 'label="%s",' % attr['order']
            except Exception:
                pass
            str_attr = str_attr.strip(',')
            line = line % str_attr 
        self.body += line + ";\n" 

    def write(self, filename):
        '''docstring for write''' 
        try:
            f = open(filename,'w')
            content =  '''
                digraph visualisation{ 
                    node[style = filled]
                    %s
                    }
            ''' 
            content = content % self.body
            f.write(content)
        except Exception, e:
            print e
            raise e
        finally:
            f.close()

class Viz_Graph(object):
    """ draw the graph """
    def __init__(self, viz = Dotty_Output()):
        self._nx_graph = nx.DiGraph()
        self.viz = viz
        self.no_nodes = defaultdict(int)

    def add_edge(self, source, target, **attr):
        self._nx_graph.add_edge(str(source), str(target))
        if attr:
           self.set_edge_attr(str(source), str(target), **attr) 
    # should use carefully
    def add_edge_unique(self, source, target, **attr):
        '''docstring for add_edge_unique''' 
        source = self.unique_id(source)
        target = self.unique_id(target)
        self.add_edge(source, target, **attr)

    # should use carefully
    def unique_id(self, node):
        ''' supposed to added this node later''' 
        node = str(node)
        self.no_nodes[node] += 1
        no_node = self.no_nodes[node]
        if no_node > 1:
            # have node with id source already, make it unique
            node = node + "[%s]" % str(no_node)
        return node

    def add_node(self, node_id, **attr):
        self._nx_graph.add_node(str(node_id))
        for key, value in attr.items():
            self._nx_graph.node[str(node_id)][key] = value

    def output_node(self, node_id):
        '''docstring for output_node''' 
        assert self.viz and callable(getattr(self.viz, "output_node"))
        self.viz.output_node(str(node_id))

    def output_edge(self, edge_id, source, target):
        '''docstring for output_edge''' 
        assert self.viz and callable(getattr(self.viz, "output_edge"))
        self.viz.output_edge(str(edge_id), str(source), str(target))

    def neighbors(self, node_id):
        '''return a list of node's neighbors'''  
        return self._nx_graph.neighbors(str(node_id))

    def set_node_attr(self, node_id, **attr):
        for key, value in attr.items():
            self._nx_graph.node[str(node_id)][key] = value

    def get_node_attr(self, node_id):
        return self._nx_graph.node[str(node_id)]

    def set_edge_attr(self, source, target, **attr):
        for key, value in attr.items():
            self._nx_graph[str(source)][str(target)][key] = value

    def get_edge_attr(self, source, target):
        #return self._nx_graph.node[id]
        return self._nx_graph[str(source)][str(target)]
    def number_of_nodes(self):
        '''docstring for number_of_nodes''' 
        return self._nx_graph.number_of_nodes()
    
    def write(self, filename):
        """docstring for write_dot"""
        assert self.viz 
        # output nodes
        for node in self._nx_graph.nodes():
            attr_dict = self._nx_graph.node[str(node)]
            self.viz.output_node(node, attr_dict)

        # output edges
        for edge in self._nx_graph.edges():
            attr_dict = self._nx_graph.edge[edge[0]][edge[1]]
            print attr_dict
            self.viz.output_edge(edge[0], edge[1], attr = attr_dict)
        self.viz.write(filename)

    #def write_json(self, root,parent = None):
        #""" write to a javascript library"""
        #data = '''{
                    #"id": "%s",
                    #"name" : "%s",
                    #"data": {
                    #"band": "%s",
                    #"relation": "member of band" 
                    #},
                    #"children":[%s] } '''
        #children = "" 
        #for child in self._nx_graph.neighbors(root):
            #str_child = self.write_json(child,root)
            #if str_child:
                #temp = "%s," %(str_child)
                #children += temp
        #if children:
            #children = children[0:len(children) - 1 ]
        #return data %(root,root,parent,children)

    def clear(self):
        """docstring for clear"""
        self._nx_graph.clear()

class Graph_Abserver(object):
    """ abstract class that help to abserve the graph according to the given filter imfo"""
    def __init__(self, source, e_types, n_types, inheritance = True):
        self.source = source
        self.valid_edge_types = e_types
        self.valid_node_types = n_types
        self.graph = Viz_Graph()
        self.inheritance = inheritance


    def write(self, filename):
        '''docstring for write_dot''' 
        self.graph.write(filename)

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
                        self.graph.add_edge(str(nodes[0]), str(nodes[1]))
                        # add edge attribute

    # abstract fuctions must be institated
    def graph_info(self):
        '''docstring for graph_info''' 
        #nodes = { }
        #for e_type in self.valid_edge_types:
            #edges = self._get_edges(e_type)
            #for edge in edges:
                #nodes = self._nodes_from_edge(edge)
                ## none empty edges!
                #if len(nodes) > 0:
                    #if self.valid_edge(edge,nodes):
                        #self.edge_types.setdefault(self._edge_type(edge), 0)
                        #self.edge_types[self._edge_type(edge)] += 1
                        #nodes[self._node_type(nodes[0])].append(nodes[0].)
                        
                        #self.graph.add_edge(nodes[0], nodes[1])
        pass

    def _nodes_from_edge(self,edge):
        pass

    def _get_edges(self,e_type):
        '''type of e_type is consistency with valid_edge_types '''
        pass
    def _edge_type(self,edge):
        '''type of edge is consistency with valid_edge_types '''
        return edge.type

    def _node_type(self,node):
        '''type node is consistency with valid_node_types '''
        return node.type
    # end 
    def _edge_is_a(self, source, target):
        '''type of source and target is consistency with valid_edge_types '''
        return source == target

    def _node_is_a(self, source, target):
        '''type of source and target is consistency with valid_node_types '''
        return source == target


    def valid_edge(self,edge,nodes):
	"""make sure the type edge and it targets are required type,
       if one of the target is invalid, then the edge is invalid
    """
        assert len(self.valid_edge_types) and len(self.valid_node_types) > 0
        #import pdb
        #pdb.set_trace()
        for arg in self.valid_edge_types:
            if self._edge_is_a(self._edge_type(edge), arg):
                break
        else:
            # invalid edge
            return False
        #determine if the outs of link is required node type
        for node in nodes:
            for arg in self.valid_node_types:
                if self._node_is_a(self._node_type(node), arg):
                    break
            else:
                # invalid node
                return False
        return True

