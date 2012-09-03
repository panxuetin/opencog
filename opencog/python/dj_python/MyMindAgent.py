import opencog.cogserver
from opencog.atomspace import types
class MyMindAgent(opencog.cogserver.MindAgent):
    def run(self,atomspace):
        print "9999999999999"
        atomspace.add_node(types.ConceptNode,"dingjie")
