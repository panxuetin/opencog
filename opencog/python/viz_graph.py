
import networkx as nx
from types_inheritance import type_to_name
class FakeAtom(object):
    """docstring for FakeAtom"""
    def __init__(self, type, name, tv = None, av = None):
        self.type = type
        self.name = name
        self.tv = tv
        self.av = av
        # @@ could just use attribute method
        self.type_name = type_to_name[type]         # a string
        
class Viz_Graph(object):
    """ draw the graph """
    def __init__(self):
        self._nx_graph = nx.DiGraph()

    def add_edge(self, source, target, **attr):
        self._nx_graph.add_edge(source,target)

    def add_node(self, node, **attr):
        self._nx_graph.add_node(node)
        for key, value in attr.items():
            self._nx_graph.node[node][key] = value
    def neighbors(self, node):
        '''return a list of node's neighbors'''  
        return self._nx_graph.neighbors(node)

    def set_node_attr(self, id, **attr):
        for key, value in attr.items():
            self._nx_graph.node[id][key] = value

    def get_node_attr(self, id):
        return self._nx_graph.node[id]

    def set_edge_attr(self, source, target, **attr):
        for key, value in attr.items():
            self._nx_graph[source][target][key] = value

    def get_edge_attr(self, source, target):
        #return self._nx_graph.node[id]
        return self._nx_graph[source][target]
    def number_of_nodes(self):
        '''docstring for number_of_nodes''' 
        return self._nx_graph.number_of_nodes()
    
    def write_dot(self, filename):
        """docstring for write_dot"""
        try:
            #nx.write_dot(self._nx_graph, 'tempfile')
            # try to make the graph more readable
            ff = open(filename,'w')
            content =  '''
                digraph visualisation{ 
                    node[style = filled]
                    %s
                    }
            ''' 
            body = "" 
            # output nodes
            for node in self._nx_graph.nodes():

                line =  '"%s" '% str(node) 
                attr_dict = self._nx_graph.node[node]
                if attr_dict:
                    line += "[%s]" 
                    attr = "" 
                    try:
                        attr += "color=%s," % attr_dict['color']
                    except Exception:
                        pass
                    try:
                        attr += "shape=%s," % attr_dict['shape']
                    except Exception:
                        pass
                    try:
                        attr += "style=%s," % attr_dict['style']
                    except Exception:
                        pass
                    attr = attr.strip(',')
                    line = line % attr
                body += line + ";\n" 
            # output edges
            for edge in self._nx_graph.edges():
                line =  '"%s" -> "%s" ' %(edge[0],edge[1])
                attr_dict = self._nx_graph.edge[edge[0]][edge[1]]
                if attr_dict:
                    line += "[%s]" 
                    attr = "" 
                    try:
                        attr += "color=%s," % attr_dict['color']
                    except Exception:
                        pass
                    try:
                        attr += "shape=%s," % attr_dict['shape']
                    except Exception:
                        pass
                    try:
                        attr += "style=%s," % attr_dict['style']
                    except Exception:
                        pass
                    try:
                        attr += 'label="%s",' % attr_dict['order']
                    except Exception:
                        pass
                    attr = attr.strip(',')
                    line = line % attr
                body += line + ";\n" 
            content = content % body
            ff.write(content)
        except Exception, e:
            print e
            raise e
        finally:
            ff.close()
    def write_json(self, root,parent = None):
        """docstring for write_json"""
        data = '''{
                    "id": "%s",
                    "name" : "%s",
                    "data": {
                    "band": "%s",
                    "relation": "member of band" 
                    },
                    "children":[%s] } '''
        children = "" 
        for child in self._nx_graph.neighbors(root):
            str_child = self.write_json(child,root)
            if str_child:
                temp = "%s," %(str_child)
                children += temp
        if children:
            children = children[0:len(children) - 1 ]
        return data %(root,root,parent,children)

    def clear(self):
        """docstring for clear"""
        self._nx_graph.clear()

