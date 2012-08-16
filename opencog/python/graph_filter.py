
try:
    from opencog.atomspace import AtomSpace, types, Atom, TruthValue
    import opencog.cogserver
except ImportError:
    from atomspace_remote import AtomSpace, types, Atom, TruthValue
from tree import *
from util import *
import m_util
import m_adaptors

from pprint import pprint
from collections import defaultdict
from viz_graph import tree_to_viz_graphic, Viz_Graph, trees_to_forest
import viz_graph
# to debug within the cogserver, try these, inside the relevant function:
#import code; code.interact(local=locals())
#import ipdb; ipdb.set_trace()

t = types

class ForestExtractor:
    """Extracts a forest of trees, where each tree is a Link (and children) that are true in the AtomSpace.
    The trees may share some of the same Nodes. This is used as a preprocessor for Fishgram. It makes a huge
    difference to the efficiency. There are lots of filters that let you control exactly which atoms will be used."""
    def __init__(self, atomspace):
        self.a = atomspace
        
        
        # Spatial relations are useful, but cause a big combinatorial explosion
        # @@!
        self.unwanted_atoms = set(['proximity', 'next', 'near',
            'beside', 'left_of', 'right_of', 'far', 'behind', 'in_front_of',
            'between', 'touching', 'outside', 'above', 'below', # 'inside',
            # Useless stuff. null means the object class isn't specified (something that was used in the
            # Multiverse world but not in the Unity world. Maybe it should be?
            'is_movable', 'is_noisy', 'null', 'id_null', 'Object',
            'exist', # 'decreased','increased',
            # Not useful e.g. because they contain numbers
            "AGISIM_rotation", "AGISIM_position", "AGISIM_velocity", "SpaceMap", "inside_pet_fov", 'turn', 'walk',
            'move:actor', 'is_moving', 
            # These ones make it ignore physiological feelings; it'll only care about the corresponding DemandGoals
            'pee_urgency', 'poo_urgency', 'energy', 'fitness', 'thirst',
            # These might be part of the old embodiment system or part of Psi, I'm not sure
            'happiness','sadness','fear','excitement','anger',
            'night',
            'actionFailed','decreased',
            'food_bowl', # redundant with is_edible
            #'foodState','egg','dish',
            # The can_do predicate is useful but should be in an AtTimeLink
            'can_do'])
        
        # state
        # @@!only useful for pathfinding visualization!
        # related tree shape which include pathfinding related atoms(which are replaced by variables)
        # not unique whether if there is a bug in atomspace
        self.trees = []
        # root(link) of related trees
        self.root_of_trees = []

        # path finding related atoms of a tree (a list to list), tuple to self.trees
        self.bindings = []
        # TimeNode in elements of self.bindings
        self.all_timestamps = set()
        # none TimeNode in elements of self.bindings
        self.all_objects  = set()
        # 
        # @@! 
        # variable counter
        # NOTE: If you set it to 0 here, it will give unique variables to every tree. BUT it will then count every occurrence of
        # a tree as different (because of the different variables!)
        # use this number seq to replace related atoms in trees
        #self.i = 0

        # substitution of trees which rooted "AtTimeLink" 
        # { tree: [{var: atom, ...}, ...], ...}
        # binding groups in a specfic tree is not unique!
        self.event_embeddings = defaultdict(list)
        
        # map from unique tree to set of embeddings.
        # An @embedding is a set of bindings. Maybe store the corresponding link too.
        # @@!
        # substitution of trees which rooted "AtTimeLink"
        self.tree_embeddings = defaultdict(list)
        
        # The incoming links (or rather trees/predicates) for each object.
        # For each object, a mapping from rel -> every embedding involving that object
        # {obj1:{tree1:[binding_group1, binding_group2, ...], tree2:[], ... }, obj2:{...}}
        # trees in self.trees_embeddings
        # binding groups in a specfic tree is unique
        self.incoming = defaultdict(lambda:defaultdict(set))

    class UnwantedAtomException(Exception):
        pass

    ##
    # @brief :extract a tree root at atom, some replacement happen here
    #         numbers of atom in the tree is more than which in objects
    #
    # @param atom : the root of the tree
    # @param objects :the other return value
    #
    # @return :1) tree leaf node of type num(variable) or node atom,
    #          2) tree rooted at link.type_name
    def extractTree(self,  atom, objects):
        if not self.include_atom(atom):
            raise self.UnwantedAtomException
        elif self.is_object(atom):
            # used for path finding
            objects.append(Tree(atom))
            self.i+=1
            # @@! used variable replace some type of atom
            return Var(self.i-1)
        elif self.is_action_instance(atom):
            #print 'is_action_instance', atom
            # this is moderatly tacky, but doing anything different would require lots of changes...
            # @@! used empty ListLink replace action atom
            return Tree('ListLink', [])
        elif atom.is_node():
            return Tree(atom)
        else:
            args = [self.extractTree(x,  objects) for x in atom.out]
            return Tree(atom.type_name, args)

    def extractForest(self):
        # TODO >0.5 for a fuzzy link means it's true, but probabilistic links may work differently        
        # @@!
        initial_links = [x for x in self.a.get_atoms_by_type(t.Link) if (x.tv.mean > 0.5 and x.tv.confidence > 0)]
        
        for link in initial_links:
            ##  @@! filter out some type of links as the root of a tree 
            if not self.include_tree(link): continue
            # pathfinding related atoms
            objects = []            
            self.i = 0
            try:

                # 1) tree leaf node of type Var(int tree node) or node atom, inner node as string
                # 2) tree rooted at link.type_name
                tree = self.extractTree(link, objects)
                log.debug(str(tree))
            except(self.UnwantedAtomException):
                # skip the whole tree!
                continue
            ## end of filter
            
            # policy - throw out trees with no objects
            # @@!
            if len(objects):
                # @@! could got two same tree instances!
                # which make repeated embeddings
                self.trees.append(tree)
                self.root_of_trees.append(link)
                self.bindings.append(objects)

                for obj in objects:
                    if obj.get_type() != t.TimeNode:
                        self.all_objects.add(obj)
                    else:
                        self.all_timestamps.add(obj)

                # maps from  variable(var tree) to related atom
                substitution = subst_from_binding(objects)
                if tree.op == 'AtTimeLink':
                    self.event_embeddings[tree].append(substitution)
                else:
                    self.tree_embeddings[tree].append(substitution)
                    for obj in objects:
                        self.incoming[obj][tree].add(substitution)


        
        # Make all bound trees. Enables using lookup_embeddings
        # @@? integrate to above
        # restore to the original tree
        # replace Var to atom, self.all_bound_trees is actually not a tree, just like @T, as it contain atom!
        self.all_bound_trees = [subst(subst_from_binding(b), tr) for tr, b in zip(self.trees, self.bindings)]    
    
        #pprint({tr:len(embs) for (tr, embs) in self.tree_embeddings.items()})
        
        #print self.all_objects, self.all_timestamps

    def if_right(self, tree):
        '''docstring for if_right''' 
        return tree in self.all_bound_trees

    def data_after_filter(self):
        log.info("***************************tree and bindings*******************************************" )
        for item in self.event_embeddings.items():
            log.pprint(item)
        for item in self.tree_embeddings.items():
            log.pprint(item)
        log.info("***************************tree instance*************************************************")
        for tree in self.all_bound_trees:
            log.info(str(tree))
        log.flush()

    def is_object(self, atom):
        ''' return :if atom is a specfic to pathfinding '''
        # only useful for pathfinding visualization!
        #return atom.name.startswith('at ')
        return atom.is_a(t.ObjectNode) or atom.is_a(t.SemeNode) or atom.is_a(t.TimeNode) or atom.name.startswith('at ') # or self.is_action_instance(atom)# or self.is_action_element(atom)
        
    def is_action_instance(self, atom):        
        return atom.t == t.ConceptNode and len(atom.name) and atom.name[-1].isdigit()
        
    
    #def object_label(self,  atom):
        #return 'some_'+atom.type_name

    def include_atom(self,  atom):
        """Whether to include a given atom in the results. If it is not included, all trees containing it will be ignored as well."""
        if atom.is_node():
            if (atom.name in self.unwanted_atoms or atom.name.startswith('id_CHUNK') or
                atom.name.endswith('Stimulus') or atom.name.endswith('Modulator') or
                atom.is_a(t.VariableNode)):
                return False
        else:            
            if any([atom.is_a(ty) for ty in 
                    [t.SimultaneousEquivalenceLink, t.SimilarityLink, # t.ImplicationLink,
                     t.ReferenceLink,
                     t.ForAllLink, t.AverageLink, t.PredictiveImplicationLink] ]):
                return False

        return True
    
    def include_tree(self,  link):
        """Whether to make a separate tree corresponding to this link. If you don't, links in its outgoing set can
        still get their own trees."""
#        if not link.is_a(t.SequentialAndLink):
#            return False

        ## Policy: Only do objects not times
        #if link.is_a(t.AtTimeLink):
        #    return False
        # TODO check the TruthValue the same way as you would for other links.
        # work around hacks in other modules
        # @@! filter out some type of links as the root of a tree
        if any([i.is_a(t.AtTimeLink) for i in link.incoming]):
            return False
        if link.is_a(t.ExecutionLink) or link.is_a(t.ForAllLink) or link.is_a(t.AndLink):
            return False        
        if link.is_a(t.AtTimeLink) and self.is_action_instance(link.out[1]):
            return False

        return True

    # tr = fish.forest.all_trees[0]
    # fish.forest.lookup_embeddings((tr,))
    def lookup_embeddings(self, conj):
        """Given a conjunction, do a naive search for all embeddings. Fishgram usually finds the embeddings as part of the search,
        which is probably more efficient. But this is simpler and guaranteed to be correct. So it is useful for testing and performance comparison.
        It could also be used to find (the embeddings for) extra conjunctions that fishgram has skipped
        (which would be useful for the calculations used when finding implication rules)."""

        return self.lookup_embeddings_helper(conj, (), {}, self.all_bound_trees)

    def lookup_embeddings_helper(self, conj, bound_conj_so_far, s, all_bound_trees):

        if len(conj) == 0:
            return [s]

        # Find all compatible embeddings. Then commit to that tree
        tr = conj[0]

        ret = []
        substs = []
        matching_bound_trees = []

        for bound_tr in all_bound_trees:
            s2 = unify(tr, bound_tr, s)
            if s2 != None:
                #s2_notimes = { var:obj for (var,obj) in s2.items() if obj.get_type() != t.TimeNode }
                substs.append( s2 )
                matching_bound_trees.append(bound_tr)

        for s2, bound_tr in zip(substs, matching_bound_trees):
            bc = bound_conj_so_far + ( bound_tr , )
            later = self.lookup_embeddings_helper(conj[1:], bc, s2, all_bound_trees)
            # Add the (complete) substitutions from lower down in the recursive search,
            # but only if they are not duplicates.
            # TODO I wonder why the duplication happens?
            for final_s in later:
                if final_s not in ret:
                    ret.append(final_s)
        
        return ret

    def forest_to_graph(self):
        '''docstring for forest_to_graph''' 
        graph = Viz_Graph()
        viz_trees = map(m_adaptors.Viz_OpenCog_Tree_Adaptor, self.trees)
        forest = viz_graph.Tree("forest", viz_trees)
        tree_to_viz_graphic(forest, graph)
        return graph

    #def forest_to_graph(self):
        #'''docstring for forest_to_graph''' 
        #a = AtomSpace()
        ##graph = viz_graph.Viz_Graph()
        ##viz_trees = map(m_adaptors.Viz_OpenCog_Tree_Adaptor, self.all_bound_trees)
        ##forest = viz_graph.Tree("forest", viz_trees)
        ##viz_graph.tree_to_viz_graphic(forest, graph)
        ##graph.write("forest.dot")
        ##graph.clear()
        ##import ipdb
        ##ipdb.set_trace()
        #try:
            #for tree in self.all_bound_trees:
                #log.debug(str(tree))
                ##add_tree_to_atomspace(tree,a)
            ##from atomspace_abserver import Atomspace_Abserver
            ##print " abserver...." 
            ##abserver = Atomspace_Abserver(a)
            ##abserver.graph_info()
            ##print " filter_graph...." 
            ##graph = abserver.filter_graph()
            ##print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^$$^" 
            ##num = 0
            ##for node in graph._nx_graph.nodes():
                ##if node.startswith("EvaluationLink"):
                    ##num += 1
                    ##print node
            ##print num
            ##print " writing dot" 
            ##abserver.write("atomspace.dot")
        #except Exception, e:
            #print e

        ##return graph
