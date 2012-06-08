
import tree
import networkx as nx
from rules import Rule
from util import *
from sets import Set
import pdb
#from opencog.atomspace import TruthValue

def analyze(chainer):
    """docstring for analyze"""
    infer = Inference_Analyze()
    viz_space_graph = Viz_Graph()
    chainer.results[0]
    infer.dsp_search_space(chainer.expr2pdn(chainer.results[0]),viz_space_graph)
    viz_space_graph.set_node_attr(str(chainer.expr2pdn(chainer.results[0])),color = "black")
    viz_space_graph.write_dot('space_result.dot')

    viz_path_graph = Viz_Graph()
    temp = []
    for key in chainer.trace.paths:
        t = chainer.trace.paths[key]
        t.reverse()
        # todo
        if len(t) > 0:
            temp.append(t)
    sub_graph = Viz_Graph()
    infer.dsp_algorithm_path( viz_path_graph,sub_graph,chainer.expr2pdn(chainer.results[0]),{},None, temp,0,[], True,False,True)
    viz_path_graph.set_node_attr(str(chainer.expr2pdn(chainer.results[0])),color = "black")
    viz_path_graph.write_dot('path_result.dot')
    sub_graph.write_dot("sub_graph.dot")

    temp =  viz_path_graph.write_json(str(chainer.expr2pdn(chainer.results[0])),"target0" )
    data = '''{
                "id": "target0",
                "name" : "target0",
                "data": [],
                "children":[%s] } ''' %(temp)
    #print data

    infer.print_pd(chainer.pd)
    infer.dsp_search_entries(chainer.pd)
    infer.print_axioms()
    infer.print_path(chainer.results, chainer.trace.paths, chainer.trace.step_count)
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

        

class Inference_Analyze(object):
    """docstring for Inference_Analyze"""
    def __init__(self):
        # type of rule
        self.related_axioms = Set()
        self.related_rules = Set()
        self.node_no = 0
        self.multiple_nodes = []

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
    # @brief :draw links between dag and it's args 
    #
    # @param dag
    # @param dag_no_dict :map from node to the number of apperearance of the node
    # @param viz_graph
    # @param which_paths :index of paths that potentially include current search trigger
    # @param simplify : if simplify the graph, making it more readable
    # @param proof_node: if display proof node of app
    #
    # @return 
    def dsp_algorithm_path(self,viz_graph,sub_graph,dag, dag_no_dict, dag_id,  paths, level, which_paths,
                               in_path, simplify = False, proof_node = False):
        '''dag : DAG node '''

        # display prenode to dag in the path
        # record useful rules
        if isinstance(dag.op,Rule) and dag.trace.made_by_rule:
           self.related_rules.add(dag.trace.path_pre)
        if dag_id == None:
           dag_id = str(dag) 
        #orig_paths = which_paths
        for arg in dag.args:
            # arg.tv.count > 0 restrict the app
            if  True:
                arg_which_paths = list(which_paths[0:len(which_paths)])
                arg_level = level
                # rename the node which help to display correct in graphviz
                arg_id = str(arg)
                new_arg_id = None
                self.node_no += 1
                try:
                    dag_no_dict[arg_id] += 1
                except Exception:
                    dag_no_dict[arg_id] = 1
                if isinstance(arg.op, Rule):
                    # apps
                    if simplify:
                        new_arg_id = "<%s>[%s](%s)"%(arg.trace.visit_order, dag_no_dict[arg_id],self.node_no)
                    else:
                        new_arg_id = arg_id + "<%s>[%s](%s)"%(arg.trace.visit_order, dag_no_dict[arg_id] ,self.node_no)
                else:
                    # goals
                    new_arg_id = arg_id + "[%s](%s)" % (dag_no_dict[arg_id],self.node_no)
                # add nodes and edges to tree
                viz_graph.add_edge(dag_id ,new_arg_id)
                #is_trigger = False
                ## apps
                if isinstance(arg.op, Rule):
                    ## arg is none axiom app
                    if arg.op.goals:
                        viz_graph.set_node_attr(new_arg_id, shape = "box" )
                        # @@! assume axiom could not be an inner node
                        ## marking the trigger
                        #if orig_paths == []:
                        ## first time
                            #for i,path in enumerate(paths):
                                #if path[0] == arg.op:  
                                     #which_paths = []
                                     #which_paths.append(i)
                        #arg_which_paths = []
                        #for index in which_paths:
                            ## reduce matched paths
                            #try:
                                #if  paths[index][arg_level] == arg.op:
                                    #arg_which_paths.append(index)
                                    #print arg.op
                            #except Exception:
                                ## the path don't have length as long as arg_level 
                                #pass
                            #if len(paths[index])-1 == level and paths[index][level] == arg.op:
                               #is_trigger = True
                          ## end marking the trigger
                    ## arg is axiom
                    else:
                        self.related_axioms.add(arg.op)
                    ## arg is an axiom or true app, but may not in path
                    if arg.tv.count > 0:
                        if not arg.op.goals:
                            # an axiom
                            viz_graph.set_node_attr(new_arg_id, shape = "septagon")
                        else:
                            # an true app, it's head may be an inference fact or existent fact(axiom) 
                            pass

                else:
                    ## arg is a tree
                    arg_level = level + 1
                if in_path and (arg.tv.count > 0 or type(arg.op) == tree.Tree):
                ## the  arg is in the path to target
                    # mark the prove node
                    generalized_app = dag
                    while  proof_node and generalized_app.trace.path_pre:
                            # mark in viz_graph
                            temp = generalized_app
                            generalized_app = generalized_app.trace.path_pre

                            target_id = dag_id if dag == temp else str(temp)
                            node_color = "yellow"  if temp.trace.made_by_rule else "blue" 
                            viz_graph.add_edge(str(generalized_app),target_id)
                            viz_graph.set_node_attr(str(generalized_app), color = node_color, shape = "box")
                            viz_graph.set_edge_attr(str(generalized_app),target_id, style = "dashed", color = node_color)
                            # mark in sub_graph

                            sub_graph.add_edge(str(generalized_app),target_id)
                            sub_graph.set_node_attr(str(generalized_app), color = node_color, shape = "box")
                            sub_graph.set_edge_attr(str(generalized_app),target_id, style = "dashed", color = node_color)

                    viz_graph.set_edge_attr(dag_id, new_arg_id, color = "red", weight = 0.5 )
                    viz_graph.set_node_attr(new_arg_id, color = "red")
                    sub_graph.add_edge(dag_id ,new_arg_id)
                    sub_graph.set_node_attr(new_arg_id, no = self.node_no)
                    if type(arg.op) == tree.Tree:
                        # tree
                        # inferenced or existent fact
                        if len([ son for son in arg.args if son.tv.count > 0]) > 1:
                            viz_graph.set_node_attr(new_arg_id, color = "yellow")
                            sub_graph.set_node_attr(new_arg_id, color = "yellow")
                            print self.node_no
                            self.multiple_nodes.append(self.node_no)
                    elif arg.op.goals:
                        # none axiom apps
                        sub_graph.set_node_attr(new_arg_id, shape = "box" )
                    else:
                        # axioms
                        sub_graph.set_node_attr(new_arg_id, shape = "septagon")
                    self.dsp_algorithm_path(viz_graph,sub_graph,arg, dag_no_dict, new_arg_id, paths,arg_level,arg_which_paths, True, simplify, proof_node)
                            
                else:
                ## not in the right path 
                    viz_graph.set_edge_attr(dag_id, new_arg_id, color = "green", weight = 0.5 )
                    self.dsp_algorithm_path(viz_graph,sub_graph, arg, dag_no_dict, new_arg_id, paths,arg_level,arg_which_paths, False, simplify, proof_node)
                    viz_graph.set_node_attr(new_arg_id, shape = "point")
                #if is_trigger:
                   #viz_graph.set_node_attr(new_arg_id, color = "yellow") 


    def print_pd(self,searched_items):
        """ print logic.pd"""
        log.info(format_log(0, False, "********************self.pd***********************" ))
        for dag in searched_items:
            log.info(format_log(0, False,dag))
    ## @todo yeild
    def dsp_search_entries(self,searched_items):
        log.info(format_log(0, False, "******************** elements in DAG ***************" ))
        """ print all elements in DAG structure"""
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
            log.info(format_log(0, False,unique_dag))
    def print_axioms(self):

        ''' print all axioms used '''
        log.info(format_log(0, False, "******************* related axioms *****************" ))
        for axiom in self.related_axioms:
            log.info(format_log(0, False, str(axiom)))
        log.info(format_log(0, False, "******************* related rules *****************" ))
        for rule in self.related_rules:
            log.info(format_log(0, False, str(rule)))
    def print_path(self, results, paths, step_count):
        """docstring for print_path"""
        log.info(format_log(0,False,"***************** results :*************************"))
        for expand in results:
            log.info(format_log(1,False, str(expand)))
        #for app in begin_apps:
            #log.info(format_log(0, True, str(app)))
            #log.ident = 0
            #while app.trace.path_pre:
                #app = app.trace.path_pre
                #log.ident += 3
                #log.info(format_log(log.ident, True, str(app)))
            #log.ident = 0
        #for path in paths:
            #log.error(format_log(0, False, path))
        for key in paths:
            log.info("****************** paths ******************" )
            log.ident = 0
            for app in paths[key]:
                log.ident += 3
                log.info(format_log(log.ident, False, str(app)))

        log.info("****************** step count :*************************")
        log.info(format_log(1,False,"bc_step:" + str(step_count)))
        log.info("****************** number of nodes in search tree :*************************")
        # plus the target node( + 1)
        log.info(format_log(1, False, str(self.node_no + 1)))
        log.info("****************** no of multiple nodes :*************************")
        log.info(format_log(1, False, str(self.multiple_nodes)))

