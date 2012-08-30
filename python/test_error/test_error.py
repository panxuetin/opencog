from opencog.atomspace import types
from m_util import Logger
log = Logger("test_error.log")
log.add_level(Logger.DEBUG)
def run_test(a):
    a.add_link(types.InheritanceLink,[a.add_node(types.ConceptNode,"DdD"),a.add_node(types.ConceptNode,"BbB")])
    a.add_link(types.InheritanceLink,[a.add_node(types.ConceptNode,"DdD"),a.add_node(types.ConceptNode,"BbB")])
    for link in a.get_atoms_by_type(types.InheritanceLink):
        log.debug(str(link)+"-------" +str(link.h.value()))
        for node in link.out:
            log.debug(str(node.h.value()))
    log.flush()
