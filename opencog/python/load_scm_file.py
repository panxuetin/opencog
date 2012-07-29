from viz_graph import Viz_Graph, FakeAtom
from types_inheritance import name_to_type, is_a
from opencog.atomspace import AtomSpace, Atom, types, TruthValue 
from fishgram import Fishgram, notice_changes
import fileinput                         
import networkx as nx
import atomspace_abserver
import m_util
a = AtomSpace();

            
def add_tree_to_atomspace(tree, root):
     ''' add nodes of a tree to atomspace in postorder ''' 
     global a
     out = []
     fakeatom = tree.get_node_attr(root)['atom']
     if not tree.neighbors(root):
     ## a leaf in the tree
         try:
            if is_a(fakeatom.type, types.Node):
                # a node
                return a.add_node(fakeatom.type, fakeatom.name, fakeatom.tv).h
            else:
                # empty link
                return a.add_link(fakeatom.type, [], fakeatom.tv).h
         except Exception, e:
             print " **** error occurs when adding to atomspace ****" 
             print e

     ## an inner node in the tree(a link), constructing it's out 
     ## ordering children as the scheme file, as there are no paticular order in the @G.neighbors function
     order_child = { }
     children = []
     for child in tree.neighbors(root):
         order_child[tree.get_edge_attr(root,child)['order']] = child
     for order in sorted(order_child):
         children.append(order_child[order])
     ## 
     for child in children:
         out.append(add_tree_to_atomspace(tree, child))
     ## construct the link
     #print "adding %s + "%root + str(out)
     return a.add_link(fakeatom.type, out, fakeatom.tv).h

def scheme_file_to_atomspace():
    tree = Viz_Graph()
    ident_stack = []
    atom_stack = []
    root = None
    no_link = { }
    define_dict = { }
    for line in fileinput.input(inplace = False):
        ## parase scheme file line by line
        ## parase "define" 
        temp = line.strip('\n ()')
        if temp.startswith('define'):
            temp = temp.split(' ')
            define_dict[temp[1]] = temp[2]
            continue
        ##
        if line.startswith('('):
            if tree.number_of_nodes() > 0:
            # deal with previous segment
                add_tree_to_atomspace(tree, root)
                tree.clear()
                no_link.clear()
            # 
            ident_stack[:] = []
            atom_stack[:] = []
        ## parase a new segment
        name = "" 
        t = "" 
        stv = None
        av = {}
        ident = line.find("(") 
        if ident != -1:
            ident_stack.append(ident)
            # the first: type name
            # the second: stv or av
            # the third: stv or av
            l = line
            line = line.strip(' ')
            line = line.strip('\n')
            elms = line.split('(')
            first = elms[1]
            try:
                second = elms[2]
                second = second.split(')')[0]
                second.replace
                second = m_util.replace_with_dict(second, define_dict)
            except Exception:
                second = None

            try:
                third = elms[3]
                third = third.split(')')[0]
                third = m_util.replace_with_dict(third, define_dict)
            except Exception:
                third = None

            if second:
                second = second.strip()
                if second.find("av") != -1:
                    temp = second.split(" ")
                    av['sti'] = float(temp[1])
                    av['lti'] = float(temp[2])
                    av['vlti'] = float(temp[3])
                else:
                    temp = second.split(" ")
                    count = float(temp[1])
                    strenth = float(temp[2])
                    stv = TruthValue(strenth, count)

            if third:
                third = third.strip()
                if third.find("av") != -1:
                    temp = third.split(" ")
                    av['sti'] = float(temp[1])
                    av['lti'] = float(temp[2])
                    av['vlti'] = float(temp[3])
                else:
                    temp = third.split(" ")
                    count = float(temp[1])
                    strenth = float(temp[2])
                    stv = TruthValue(strenth, count)
            try:
                first.index(' ')
                temp  =  first.split(' ')
                t = temp[0].strip(' ')
                name = temp[1].split('"')[1]
            except Exception:
                t = first.strip(' ')

            ## add nodes to segment tree
            if name != "" :
                # node
                try:
                    node = FakeAtom(name_to_type[t], name, stv, av)
                except Exception, e:
                    print "Unknown Atom type '%s' in line %s"% (t,fileinput.lineno())
                    raise e
                tree.add_node(name, atom = node)
                atom_stack.append(node)
                if l.startswith('('):
                    root = name
            else:
                # link
                no = no_link.setdefault(t,1)
                no_link[t] += 1
                link_name = t + str(no)
                try:
                    link = FakeAtom(name_to_type[t], link_name, stv, av)
                except Exception, e:
                    print "Unknown Atom type '%s' in line %s"% (t,fileinput.lineno())
                    raise e
                atom_stack.append(link)
                tree.add_node(link_name, atom = link)
                if l.startswith('('):
                    root = link_name

            ## add an edge to the segment tree
            now = ident_stack[-1]
            for i, prev_ident in reversed(list(enumerate(ident_stack))):
                if now > prev_ident:
                    ## the ith is parent
                    #print atom_stack[i].name + "->" + atom_stack[-1].name
                    tree.add_edge(atom_stack[i].name, atom_stack[-1].name)
                    ## set the 'order' attribute
                    try:
                        tree.get_node_attr(atom_stack[i].name)['order'] += 1
                    except Exception:
                        tree.get_node_attr(atom_stack[i].name)['order'] = 0
                    order = tree.get_node_attr(atom_stack[i].name)['order']
                    #print atom_stack[-1].name + str(order)
                    tree.set_edge_attr(atom_stack[i].name, atom_stack[-1].name, order = order)
                    break

    ## deal with the last segment
    if tree.number_of_nodes() > 0:
        add_tree_to_atomspace(tree, root)
        tree.clear()
        no_link.clear()
    print "loading sucessfully!" 
    return a
scheme_file_to_atomspace()
fish = Fishgram(a)


# Detect timestamps where a DemandGoal got satisfied or frustrated
notice_changes(a)

fish.forest.extractForest()
print (fish.forest.all_trees)
print "**************************************************************************************" 
fish.run()
print "Finished one Fishgram cycle"
#abserver = atomspace_abserver.Atomspace_Abserver(a)
#abserver.add_valid_edges()
#abserver.write_dot("haha.dot")
#for link in a.get_atoms_by_type(types.InheritanceLink):
    #print "&&&&&&&&&&&&&&&&&&&&&&&&&&" 
    #for out in a.get_outgoing(link.h):
        #print a.get_atom_string(out.h)
