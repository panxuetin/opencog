import os
from opencog.atomspace import AtomSpace, Atom, types, TruthValue as tv
from tree import  *
from util import *
from logic import *
from inference_analyze import analyze
#from test_tree import TreeTest
space = AtomSpace();

truth = tv(0.5,5)
a = space.add_node(types.ConceptNode,"A",truth)
b = space.add_node(types.ConceptNode,"B",truth)
c = space.add_node(types.ConceptNode,"C",truth)
d = space.add_node(types.ConceptNode,"D",truth)
space.add_link(types.InheritanceLink,[a ,b] ,truth)
space.add_link(types.InheritanceLink,[b, c],truth)
space.add_link(types.InheritanceLink,[space.add_node(types.ConceptNode,"D"), space.add_node(types.ConceptNode,"B")],truth)
space.add_link(types.InheritanceLink,[space.add_node(types.ConceptNode,"D"), space.add_node(types.ConceptNode,"B")],truth)
space.add_link(types.InheritanceLink,[c, d],truth)
#reason = Chainer(space)
#target =  T('ImplicationLink', a, d)
target =  T('ImplicationLink', a, c)
for link in space.get_atoms_by_type(types.Link):
    print link
    # code...
#result = reason.bc(target,2000)

#print "^^^^^^^^^^^^^^^^^^^^66" 
#t = reason.expr2pdn(T(c))
#print str(t)
#for arg in t.args:
    #print str(arg)
#analyze(reason)

# print 
#log.info("*****************data in atomspace****************")
#for atom in space.get_atoms_by_type(types.Atom):
    #log.info(atom.name + ":" + atom.type_name)
#log.info("*****************data in atomspace****************")
