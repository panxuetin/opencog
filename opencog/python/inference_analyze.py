
import tree
import networkx as nx
from rules import Rule
from util import *
#from opencog.atomspace import TruthValue

def analyze(chainer):
    """docstring for analyze"""
    infer = Inference_Analyze()
    viz_space_graph = Viz_Graph()
    chainer.results[0]
    infer.dsp_search_space(chainer.expr2pdn(chainer.results[0]),viz_space_graph)
    viz_space_graph.write_dot('space_result.dot')

    viz_path_graph = Viz_Graph()
    #infer.dsp_search_path(chainer.expr2pdn(chainer.results[0]),{},{},viz_path_graph)
    #viz_path_graph.write_dot('path_result.dot')

    viz_path_graph.clear()
    infer.dsp_search_path(chainer.expr2pdn(chainer.results[0]),{ },{ }, viz_path_graph,True,True)
    viz_path_graph.write_dot('proof_path_result.dot')

    viz_path_graph.clear()
    infer.dsp_search_path(chainer.expr2pdn(chainer.results[0]),{ },{ }, viz_path_graph,True,False)
    viz_path_graph.write_dot('simplify_path_result.dot')

    viz_path_graph.clear()
    infer.dsp_search_path(chainer.expr2pdn(chainer.results[0]),{ },{ }, viz_path_graph,False,False)
    viz_path_graph.write_dot('path_result.dot')

    temp =  viz_path_graph.write_json(str(chainer.expr2pdn(chainer.results[0])),"target0" )
    data = '''{
                "id": "target0",
                "name" : "target0",
                "data": [],
                "children":[%s] } ''' %(temp)
    #print data

    #infer.print_pd(chainer.pd)
    #infer.dsp_search_entries(chainer.pd)
class Viz_Graph(object):
    """ draw the graph """
    def __init__(self):
        self._nx_graph = nx.DiGraph()

    def add_edge(self, source, target, **attr):
        self._nx_graph.add_edge(source,target)

    def add_node(self, node, **attr):
        pass

    def set_node_attr(self, id, **attr):
        for key, value in attr.items():
            self._nx_graph.node[id][key] = value

    def get_node_attr(self, id, **attr):
        return self._nx_graph.node[id]

    def set_edge_attr(self, source, target, **attr):
        for key, value in attr.items():
            self._nx_graph[source][target][key] = value

    def get_edge_attr(self, id, **attr):
        #return self._nx_graph.node[id]
        pass
    

    def write_dot(self, filename):
        """docstring for write_dot"""
        try:
            nx.write_dot(self._nx_graph, 'tempfile')
            # try to make the graph more readable
            f = open('tempfile','r')
            ff = open(filename,'w')
            lines = f.readlines()
            lines[0] = " digraph visualisation{ \n" 
            lines.insert(1,"node[style = filled]\n")
            ff.writelines(lines)
        except Exception, e:
            print e
            raise e
        finally:
            f.close()
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

        

class Inference_Analyze(object):
    """docstring for Inference_Analyze"""
    def __init__(self):
        pass

    def dsp_search_space(self, dag, viz_graph):
        '''dag : DAG node '''
        # output current node
        for parent in dag.parents:
            #self.viz.outputTreeNode(dag,parent,1)
            parent_id = str(parent)
            target_id = str(dag)
            viz_graph.add_edge(parent_id,target_id)
        # output children
        for arg in dag.args:
            self.dsp_search_space(arg,viz_graph)

    ##
    # @brief :display necessary search path that reach the target(plotting from current dag to args)
    #
    # @param dag
    # @param dic_count :map from node to the number of apperearance of the node
    # @param dic_dag   :help to get the unique name of each node, dict like " 'c':'c3' " 
    # @param viz_graph
    # @param simplify : if simplify the graph, making it more readable
    # @param proof_node: if display proof node of app
    #
    # @return 
    def dsp_search_path(self,dag, dic_count, dic_dag, viz_graph, simplify = False, proof_node = False):
        '''dag : DAG node '''
        # output current node
        
        dag_id = str(dag)
        # the map from arg_id to new_arg_id
        new_dic_dag = { }
        # display prenode to dag in the path
        if simplify and proof_node and dag.trace.path_pre:
            try:
                viz_graph.add_edge(str(dag.trace.path_axiom),dic_dag[dag_id])
                viz_graph.add_edge(str(dag.trace.path_pre),dic_dag[dag_id])
                viz_graph.set_node_attr(str(dag.trace.path_axiom), color = "red")
                viz_graph.set_node_attr(str(dag.trace.path_pre), color = "red")
                #viz_graph.set_edge_attr(s))
                viz_graph.set_edge_attr(str(dag.trace.path_axiom),dic_dag[dag_id], style = "dashed", color = "red" )
                viz_graph.set_edge_attr(str(dag.trace.path_pre),dic_dag[dag_id], style = "dashed", color = "red" )

            except Exception:
                viz_graph.add_edge(str(dag.trace.path_axiom),dag_id)
                viz_graph.add_edge(str(dag.trace.path_pre),dag_id)
                viz_graph.set_node_attr(str(dag.trace.path_axiom), color = "red")
                viz_graph.set_node_attr(str(dag.trace.path_pre), color = "red")
                viz_graph.set_edge_attr(str(dag.trace.path_axiom),dag_id, style = "dashed", color = "red" )
                viz_graph.set_edge_attr(str(dag.trace.path_pre),dag_id, style = "dashed",color = "red" )
            finally:
                if dag.trace.path_pre.trace.path_pre:
                    # print the rule
                    viz_graph.add_edge(str(dag.trace.path_pre.trace.path_pre),str(dag.trace.path_pre))
                    viz_graph.set_node_attr(str(dag.trace.path_pre.trace.path_pre), color = "yellow" )
                    viz_graph.set_edge_attr(str(dag.trace.path_pre.trace.path_pre),str(dag.trace.path_pre), style = "dashed" 
                                                                                       ,color = "red" )
        for arg in dag.args:
            if  not simplify or  type(arg.op) ==  tree.Tree or arg.tv.count > 0 :
                arg_id = str(arg)
                new_arg_id = None
                try:
                    dic_count[arg_id] += 1
                    new_arg_id = arg_id + "[%s]" % str(dic_count[arg_id])
                    new_dic_dag[arg_id] = new_arg_id
                except Exception:
                    dic_count[arg_id] = 1
                if not new_arg_id:
                   new_arg_id = arg_id 

                try:
                    viz_graph.add_edge(dic_dag[dag_id],new_arg_id)
                    if proof_node and arg.trace.path_pre:
                        viz_graph.add_edge(dic_dag[dag_id],str(arg.trace.path_pre))
                        viz_graph.set_edge_attr(str(dic_dag[dag_id]),str(arg.trace.path_pre), style = "dashed", color = "red" )
                    # dag id has been replaced by parent
                except Exception:
                    viz_graph.add_edge(dag_id,new_arg_id)
                    if proof_node and arg.trace.path_pre:
                        viz_graph.add_edge(dag_id,str(arg.trace.path_pre))
                        viz_graph.set_edge_attr(dag_id,str(arg.trace.path_pre), style = "dashed", color = "red" )
        for arg in dag.args:
            if not simplify or type(arg.op) == tree.Tree or arg.tv.count > 0 :
                self.dsp_search_path(arg, dic_count, new_dic_dag, viz_graph, simplify, proof_node)

    def print_pd(self,searched_items):
        """docstring for print_pd"""
        log.debug(format_log(0, False, "********************self.pd***********************" ))
        for dag in searched_items:
            log.debug(format_log(0, False,dag))
    ## @todo yeild
    def dsp_search_entries(self,searched_items):
        log.debug(format_log(0, False, "********************searched entries***************" ))
        """docstring for search_entries"""
        temp = tree.OrderedSet()
        def print_self_pd(dag):
            """docstring for fname"""
            if not dag.args:
               temp.append(dag) 
               return 
            for arg in dag.args:
                temp.append(arg)
                if arg.args:
                   print_self_pd(arg) 
        for entrance in searched_items:
            print_self_pd(entrance)
        for unique_dag in temp:
            log.debug(format_log(0, False,unique_dag))
