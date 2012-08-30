from opencog.atomspace import AtomSpace, Atom, types, TruthValue 
from load_scm_file import load_scm_file
from viz_graph import Viz_Graph, tree_to_viz_graphic, Dotty_Output
from m_adaptors import Viz_OpenCog_Tree_Adaptor
from atomspace_abserver import Atomspace_Abserver
from graph_filter import ForestExtractor
from tree import tree_from_atom
from m_util import log
def test_tree_to_viz_graph():
    '''docstring for test_tree_to_graph''' 
    a = AtomSpace()
    load_scm_file(a, "test.scm")
    # test @tree_to_graphic
    links = a.get_atoms_by_type(types.EvaluationLink)
    t = tree_from_atom(links[0])
    writer = Dotty_Output()
    graph = Viz_Graph(writer)
    tree_to_viz_graphic(Viz_OpenCog_Tree_Adaptor(t), graph)
    graph.write("tree_to_graphic.dot" )

    
def test_forests():
    '''docstring for test_forests''' 
    a = AtomSpace()
    load_scm_file(a, "air.scm")
    log.debug("load sucessfully" )
    forest = ForestExtractor(a)
    forest.extractForest()
    graph = forest.forest_to_graph()
    graph.write("forest.dot")

if __name__ == '__main__':
    test_forests()
