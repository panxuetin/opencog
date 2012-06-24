# You can test at the cogserver shell, using
# import adaptors; reload(adaptors); import fishgram; reload(fishgram);from fishgram import *; fish = FishgramMindAgent(); fish.run(a)
try:
    from opencog.atomspace import AtomSpace, types, Atom, TruthValue, types as t
    import opencog.cogserver
except ImportError:
    from atomspace_remote import AtomSpace, types, Atom, TruthValue, types as t
    
from tree import *
import adaptors
from pprint import pprint
from util import *
import util
from itertools import *
from collections import namedtuple, defaultdict
import sys
import time
import math

from logic import PLNviz

import gc
import sys

# unit of timestamps is 0.01 second so multiply by 100
interval = 100* 20

def format_log(*args):
    global _line    
    out = str(_line) + ' ' + ' '.join(map(str, args))
#    if _line == 39:
#        import pdb; pdb.set_trace()
    _line+=1
    return out
_line = 1

def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    >>> list(pairwise((1,2,3,4)))
    [(1, 2), (2, 3), (3, 4)]
    """
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

class Pattern:
    '''Store a basic pattern and other associated data for Fishgram.'''
    def __init__(self, conj):
        self.conj = conj
        self.seqs = ()
        self.embeddings = []
    
    def __str__(self):
        return '\x1B[1;37mPattern(\x1B[1;31m'+pp(self.conj)+' \x1B[1;34m'+pp(self.seqs)+'\x1B[1;37m)'

class Fishgram:
    def __init__(self,  atomspace):
        self.forest = adaptors.ForestExtractor(atomspace,  None)
        # settings
        self.min_embeddings = 2
        self.max_embeddings = 2000000000
        self.min_frequency = 0.5
        self.atomspace = atomspace
        
        self.max_per_layer = 100
        
        self.viz = PLNviz(atomspace)
        self.viz.connect()
        self.viz.outputTreeNode(target=[], parent=None, index=0)
        
        self.rules_output = []
        
        self._is_running = False

    def run(self):
        '''The basic way to run Fishgram. It will find all the frequent conjunctions above min_frequency.'''

        return [layer for layer in self.closed_bfs_layers()]

    def iterated_implications(self):
        """Find implications, starting with maximum support (i.e. maximum pruning in the search for
        frequent subgraphs). Then lower the support incrementally. This is like the APRIORI rule-learning algorithm
        which finds frequent implication rules by first requiring very frequent rules and then needing less frequent
        ones if it can't find any."""
        # This number could be anything
        self.min_embeddings = 77
        
        while self.min_embeddings > 0:
            print "support =", self.min_embeddings
            self.implications()
            self.min_embeddings -= 5
#        while self.min_frequency > 0.00000000001:
#            print '\n\x1B[1;32mfreq =', self.min_frequency, '\x1B[0m'
#            self.implications()
#            self.min_frequency /= 2

    #import profiling
    #@profiling.profile_func()
    def implications(self):
        '''This method will make Fishgram search for conjunctions. After it finds conjunctions of length 2,3,4 it will
        use them to create implication rules in a similar way to the APRIORI algorithm. The code uses the Python "yield"
        command, so it can start producing the rules before the search finishes. This is useful if the search (for conjunctions) is slow.'''
        layers = []
        start = time.time()
        for layer in self.closed_bfs_layers():
            
#            for conj, embs in layer:
#                print pp(conj), len(embs) #, pp(embs)
            
            layers.append(layer)
            if len(layers) >= 2:
                self.output_causal_implications_for_last_layer(layers)
            
            print "All rules produced so far:"
            for imp in self.rules_output:
                print pp(imp)
            
#            if time.time() - start > 120:
#                print 'TIMEOUT'
#                break

# breadth-first search (to make it simpler!)
# use the extension list.
# prune unclosed conjunctions.
# you only need to add extensions if they're in the closure.

    def closed_bfs_extend_layer(self, prev_layer):
        '''Just a helper function for closed_bfs_layers'''
        #next_layer_iter = self.extensions(prev_layer)
        next_layer_iter = self.extensions_simple(prev_layer)
        #return self.prune_frequency(next_layer_iter)
        #self.viz.outputTreeNode(target=[], parent=None, index=0)
        
        # This would find+store the whole layer of extensions before pruning them
        # Less efficient but may be easier to debug
        next_layer = list(next_layer_iter)

        #for (ptn, embs) in self.prune_frequency(next_layer):
        for (ptn, embs) in self.prune_surprise(next_layer):
            #print '***************', conj, len(embs)
            #self.viz.outputTreeNode(target=conj[-1], parent=conj[:-1], index=0)
            #self.viz.outputTreeNode(target=list(conj), parent=list(conj[:-1]), index=0)
            yield (ptn, embs)

    def closed_bfs_layers(self):
        '''Main function to run the breadth-first search. It yields results one layer at a time. A layer
        contains all of the conjunctions resulting from extending previous conjunctions with one extra
        tree. For some purposes it would be better to return results immediately rather than one layer at
        a time, however creating ImplicationLinks requires previous layers. Note that in the current design,
        the code will automatically add SeqAndLinks between TimeNodes whenever possible. This means the
        conjunctions in a layer may be different lengths. But only due to having more/less SeqAndLinks; they
        will have the same number of other links.'''
        #all_bindinglists = [(obj, ) for obj in self.forest.all_objects]
        #prev_layer = [((), None )]
        empty_pattern = Pattern( () )
        empty_b = [{}]
        prev_layer = [(empty_pattern, empty_b)]

        while len(prev_layer) > 0:
            # Mixing generator and list style because future results depend on previous results.
            # It's less efficient with memory but still allows returning results sooner.
            new_layer = [ptn_embs for ptn_embs in self.closed_bfs_extend_layer(prev_layer)]
            
            if len(new_layer):
                
                del new_layer[self.max_per_layer+1:]
                
                #conj_length = len(new_layer[0][0].conj)
                conj_length = set(len(pe[0].conj+pe[0].seqs) for pe in new_layer)
                #print '\x1B[1;32m# Conjunctions of size', conj_length,':', len(new_layer), 'pruned', pruned,'\x1B[0m'
                print format_log( '\x1B[1;32m# Conjunctions of size', conj_length, ':', len(new_layer), '\x1B[0m')

                for ptn, embs in new_layer:
                    print format_log(ptn, len(embs))

                yield new_layer
            
            prev_layer = new_layer

    # Helper functions for extensions_simple
    # Code to handle variables. It's not important to understand this (to understand fishgram).
    def _create_new_variables(self, tr, embeddings):
        sa_mapping = {}
        tr = standardize_apart(tr, sa_mapping)
        
        rebound_embs = []
        for s in embeddings:
            s2 = {}
            for (old_var, new_var) in sa_mapping.items():
                obj = s[old_var]
                s2[new_var] = obj
            rebound_embs.append(s2)
        
        return tr, rebound_embs

    def _map_to_existing_variables(self, prev_binding, new_binding):
        # In this binding, a variable in the tree might fit an object that is already used.
        new_vars = [var for var in new_binding if var not in prev_binding]
        remapping = {}
        new_s = dict(prev_binding)
        for var in new_vars:
            obj = new_binding[var]
            tmp = [(o, v) for (v, o) in prev_binding.items() if o == obj]
            assert len(tmp) < 2
            if len(tmp) == 1:
                _, existing_variable = tmp[0]
                remapping[var] = existing_variable
            else:
                # If it is not a redundant var, then add it to the new binding.
                new_s[var] = obj

            # Never allow links that point to the same object twice
            tmp = [(o, v) for (v, o) in new_binding.items() if o == obj]
            if len(tmp) > 1:
                return None
        
        return remapping, new_s

    def _after_existing_actions(self,prev_seqs, tr, new_embedding):
        assert isinstance(prev_seqs, tuple)
        assert isinstance(tr, Tree)
        assert isinstance(new_embedding, dict)
        assert tr.op == 'AtTimeLink'
        
        # Only add times at the end of the sequence
        newly_added_var = tr.args[0]
        newly_added_timestamp = int(new_embedding[newly_added_var].op.name)
        
        previous_latest_time_var = prev_seqs[-1].args[0]
        previous_latest_timestamp = int(new_embedding[previous_latest_time_var].op.name)
        
        if 0 < newly_added_timestamp - previous_latest_timestamp <= interval:
            return True
        
        if (newly_added_timestamp == previous_latest_timestamp and
            prev_seqs[-1] != tr):
            return True
        
        return False

    # This is the new approach to finding extensions. It works like this:
    # Start with the basic pattern/conjunction () - which means 'no criteria at all'
    # Each 'layer', it goes through the patterns in the previous layer. For each one:
    #   Look at extra single things you could add into the pattern.
    #   The new layer contains all the resulting patterns.
    # A pattern is a connected graph, that is to say, all of the expressions in it need to share variables.
    def extensions_simple(self, prev_layer):
        '''Find all patterns (and examples) that would extend the previous layer of patterns. That is, the patterns
        that include one extra constraint.'''
        
        # Not correct - it must choose variables so that new 'links' (trees) will be connected in the right place.
        # That should be done based on embeddings (i.e. add a link if some of the embeddings have it)
        
        # But wait, you can just look it up and then merge new variables that point to existing objects.
        def constructor():
            '''Just to make sure the default value is constructed separately each time'''
            return (None,[])
        conj2ptn_emblist = defaultdict( constructor )
        
        last_realtime = time.time()
        for (ptn, s) in self.find_extensions(prev_layer):
            #print '...',time.time() - last_realtime
            last_realtime = time.time()

            conj = ptn.conj + ptn.seqs
            
#            num_variables = len(get_varlist(conj))
#            if num_variables != 1:
#                continue
            
            # Check for other equivalent ones. It'd be possible to reduce them (and increase efficiency) by ordering
            # the extension of patterns. This would only work with a stable frequency measure though.
            #clones = [c for c in conj2ptn_emblist.keys()
            #           if isomorphic_conjunctions(conj, c) and c != conj]
            #if len(clones):
            #    continue

            tmp = canonical_trees(ptn.conj)
            canonical_conj = tuple(tmp) + ptn.seqs
            
            use_ordering = True
            if use_ordering:
                # A very hacky ordering system. Simply makes sure that each link added
                # to the conjunction comes after the existing ones. I'm not sure if this
                # will exclude some things appropriately. For example the < comparison
                # will compare a mixture of predicate names and variable names. Also
                # when you add two variables, it may break things too...
                if len(tmp) >= 2 and tmp[-1] < tmp[-2]: continue
            else:
                #print 'canonical_conj', canonical_conj

                # Whether this conjunction is a reordering of an existing one. Currently the
                # canonical form only makes variable names consistent, and not orders.
                is_reordering = False
                
                #import pdb; pdb.set_trace()
                perms = [tuple(canonical_trees(perm)) + ptn.seqs
                         for perm in permutations(ptn.conj)
                         ][1:]
                
                #perms = permutated_canonical_tuples(conj)[1:]
                #print '#perms', len(perms),
                for permcanon in perms:
                    if permcanon in conj2ptn_emblist:
                        is_reordering = True
                
                if is_reordering:
                    continue
            
            #print 'clonecheck time', time.time()-last_realtime, '#atoms #seqs',len(ptn.conj),len(ptn.seqs)
            
            entry=conj2ptn_emblist[canonical_conj]
            #if not len(entry[1]):
            #    print '====+>>', ptn.conj,
            #    if len(ptn.seqs):
            #        print '<++>>>', ptn.seqs
            #    else:
            #        print
            #sys.stdout.write('.')
            
            embs = entry[1]
            if s not in entry[1]:
                embs.append(s)
            conj2ptn_emblist[canonical_conj] = (ptn, embs)

            # Faster, but causes a bug.
#            canon = tuple(canonical_trees(conj))
#            print 'conj', pp(conj)
#            print 'canon', pp(canon)
#            conj2emblist[canon].append(s)
            #print 'extensions_simple', len(conj2emblist[canon])
            
        return conj2ptn_emblist.values()

    def find_extensions(self, prev_layer):
        '''Helper function for extensions_simple. It's a generator that finds all conjunctions (X,Y,Z) for (X,Y) in
        the previous layer. It returns a series of (conjunction, substitution) pairs. Where each substitution is
        one way to produce an atom(s) in the AtomSpace by replacing variables in the conjunction. The conjunctions
        will often appear more than once.'''
        
        for (prev_ptn,  prev_embeddings) in prev_layer:

            for tr_, embs_ in self.forest.tree_embeddings.items():

#                if prev_conj != () and tr_ < self.awkward[prev_conj]:
#                    #print 'OUT_OF_ORDER', tr_
#                    continue
                
                # Give the tree new variables. Rewrite the embeddings to match.
                tr, rebound_embs = self._create_new_variables(tr_, embs_)
                
                # They all have the same 'link label' (tree) but may be in different places.
                for s in rebound_embs:
                    for e in prev_embeddings:
                        # for each new var, if the object is in the previous embedding, then re-map them.
                        
                        tmp = self._map_to_existing_variables(e, s)
                        if tmp == None:
                            continue
                        remapping, new_s = tmp
                        
                        remapped_tree = subst(remapping, tr)
                        
                        if remapped_tree in prev_ptn.conj:
                            continue
                        
                        if tr_.op == 'AtTimeLink' and prev_ptn.seqs:
                            after = self._after_existing_actions(prev_ptn.seqs,remapped_tree,new_s)

                        # There needs to be a connection to the existing pattern.
                        # A connection can be one or both of: reusing a variable (for an object or timenode);
                        # or the latest action being shortly after the existing ones. The first action must
                        # be connected to an existing object, i.e. it's not after anything but there is a
                        # remapping.
                        conj = prev_ptn.conj
                        seqs = prev_ptn.seqs
                        #import pdb; pdb.set_trace()
                        
                        firstlayer = (prev_ptn.conj == () and prev_ptn.seqs == ())
                        if tr_.op != 'AtTimeLink':
                            if len(remapping) or firstlayer:
                                conj += (remapped_tree,)
                            else:
                                continue
                        else:
                            if len(prev_ptn.seqs) == 0:
                                accept = ( len(remapping) or firstlayer)
                            else:
                                # Note: 'after' means the new timestamp is greater than OR EQUAL TO the existing one.
                                # seqs will always contain an exact sequence, so you can't refer to other actions involving the
                                # same object(s) but at a different time...
                                accept = after
                            
                            if accept:
                                seqs += (remapped_tree,)
                            else:
                                continue
                        
                        #print format_log('accepting an example for:',prev_ptn,'+',remapped_tree)
                        
                        remapped_ptn = Pattern(conj)
                        remapped_ptn.seqs = seqs

                        self.viz.outputTreeNode(target=list(remapped_ptn.conj+remapped_ptn.seqs),
                                                parent=list(prev_ptn.conj+prev_ptn.seqs), index=0)

                        yield (remapped_ptn, new_s)

    def prune_frequency(self, layer):
        for (ptn, embeddings) in layer:
            #self.surprise(conj, embeddings)
            
            #import pdb; pdb.set_trace()
            count = len(embeddings)*1.0
            num_possible_objects = len(self.forest.all_objects)*1.0
            num_variables = len(get_varlist(ptn.conj))*1.0
            
            normalized_frequency =  count / num_possible_objects ** num_variables
            if len(embeddings) >= self.min_embeddings and len(embeddings) <= self.max_embeddings:
            #if normalized_frequency > self.min_frequency:
                #print pp(conj), normalized_frequency
                yield (ptn, embeddings)

    def prune_surprise(self, layer):
        for (ptn, embeddings) in layer:
            if len(embeddings) >= self.min_embeddings:
                if len(ptn.conj) + len(ptn.seqs) < 2:
                    yield (ptn, embeddings)
                else:
                    surprise = self.surprise(ptn)
                    if len(ptn.conj) > 0 and surprise > 0.9: # and len(get_varlist(ptn.conj)) == 1 and len(ptn.seqs) == 0:
                        print '\x1B[1;32m%.1f %s' % (surprise, ptn)
                        yield (ptn, embeddings)
    
    
    def surprise(self, ptn):
        conj = ptn.conj + ptn.seqs

        #print 'incremental:', embeddings
        embeddings = self.forest.lookup_embeddings(conj)
        #print 'search:', embeddings
        
        Nconj = len(embeddings)*1.0        
        Pconj = Nconj/self.total_possible_embeddings(conj,embeddings)
        #Pconj = self.num_obj_combinations(conj,embeddings)/self.total_possible_embeddings(conj,embeddings)

        #print 'P) \x1B[1;32m%.5f %s' % (Pconj, ptn)
        
        P_independent = 1
        for tr in conj:
            Etr = self.forest.lookup_embeddings((tr,))
            P_tr = len(Etr)*1.0 / self.total_possible_embeddings((tr,), Etr)
            #P_tr = self.num_obj_combinations((tr,), Etr)/self.total_possible_embeddings((tr,), Etr)
            P_independent *= P_tr

        #P_independent = P_independent ** (1.0/len(conj))

        #self.distribution(ptn, embeddings)

        surprise = Pconj / P_independent
        #print conj, surprise, P, P_independent, [Nx/N for Nx in Nxs], N
        #surprise = math.log(surprise, 2)
        return surprise
    
    def num_obj_combinations(self, conj, embeddings):
        '''Count the number of combinations of objects, such that everything in the conjunction is true,
        and there is at least one time period where all of the events happened. It's equivalent to having
        an AverageLink for the objects and ExistLink for the times.'''
        # Find every unique embedding after removing the times.
        embs_notimes = set( 
            frozenset((var,obj) for (var,obj) in emb.items() if obj.get_type() != t.TimeNode)
            for emb in embeddings)
        
        return len(embs_notimes)*1.0
    

    def total_possible_embeddings(self, conj, embeddings):
        N_objs = len(self.forest.all_objects)*1.0
        N_times = len(self.forest.all_timestamps)*1.0
        
        # The number of possible embeddings for that combination of object-variables and time-variables
        N_tuples = 1
        for var in get_varlist(conj):
            if var not in embeddings[0]:
                #print 'ERROR', conj
                return 100000000000000.0
            if embeddings[0][var].get_type() == t.TimeNode:
                #N_tuples *= N_times
                pass
            else:
                N_tuples *= N_objs
        
        return N_tuples*1.0

    
    def outputConceptNodes(self, layers):
        id = 1001
        
        for layer in layers:
            for (conj, embs) in layer:                
                if (len(get_varlist(conj)) == 1):
                    concept = self.atomspace.add_node(t.ConceptNode, 'fishgram_'+str(id))
                    id+=1
                    print concept
                    for tr in conj:
                        s = {Var(0):concept}
                        bound_tree = subst(s, tr)
                        #print bound_tree
                        print atom_from_tree(bound_tree, self.atomspace)

    def outputPredicateNodes(self, layers):
        id = 9001
        
        for layer in layers:
            for (conj, embs) in layer:
                predicate = self.atomspace.add_node(t.PredicateNode, 'fishgram_'+str(id))
                id+=1
                #print predicate
                
                vars = get_varlist(conj)
                #print [str(var) for var in vars]

                evalLink = T('EvaluationLink',
                                    predicate, 
                                    Tree('ListLink', vars))
                andLink = Tree('AndLink',
                                    conj)
                
                qLink = T('ForAllLink', 
                                Tree('ListLink', vars), 
                                T('ImplicationLink',
                                    andLink,
                                    evalLink))
                a = atom_from_tree(qLink, self.atomspace)
                
                a.tv = TruthValue(1, 10.0**9)
                count = len(embs)
                #eval_a = atom_from_tree(evalLink, self.atomspace)
                #eval_a.tv = TruthValue(1, count)
                
                print a

#                for tr in conj:
#                    s = {Tree(0):concept}
#                    bound_tree = subst(s, tr)
#                    #print bound_tree
#                    print atom_from_tree(bound_tree, self.atomspace)

    def output_causal_implications_for_last_layer(self, layers):
        if len(layers) < 2:
            return
        layer = layers[-1]
        prev_layer = layers[-2]
        for (ptn, embs) in layer:
            conj = list(ptn.conj)
            seqs = list(ptn.seqs)
            
            if len(seqs) < 2:
                continue
            
            conclusion = seqs[-1]
            other = seqs[:-1]
            assert len(other)

            # Remove all of the AtTimeLinks from inside the sequence - just leave
            # the EvaluationLinks/ExecutionLinks. The AtTimeLinks are not
            # required/allowed if you have SequentialAndLinks etc. This won't change
            # the Pattern that Fishgram is storing - Fishgram's search does need
            # the AtTimeLinks.            
            conclusion_stripped = conclusion.args[1]
            other_stripped = [attime.args[1] for attime in other]
            
            # There are several special cases to simplify the Link produced.
            
            if len(other_stripped) > 1:
                # NOTE: this won't work if some of the things are simultaneous
                initial = Tree('SequentialAndLink',other_stripped)
            else:
                initial = other_stripped[0]
            
            predimp = T     ('PredictiveImplicationLink',
                                initial,
                                conclusion_stripped
                            )
            
            if len(conj) > 0:
                imp = T('ImplicationLink',
                        Tree('AndLink', conj),
                        predimp)
                payload = imp
            else:
                payload = predimp
            
            vars = get_varlist( conj + other_stripped + [conclusion_stripped] )
            assert len(vars)
            rule = T('AverageLink',
                     T('ListLink',vars),
                     payload
                    )

            # Calculate the frequency. Looking up embeddings only works if you keep the
            # AtTimeLinks.
            premises = conj + other
            premises_embs = self.forest.lookup_embeddings(premises)
            
            freq = len(embs) / len(premises_embs)
            
            a = atom_from_tree(rule, self.atomspace)
            self.rules_output.append(rule)
            
            a.tv = TruthValue(freq, len(embs))
            print a


    def find_exists_embeddings(self, embs):
        if not len(embs):
            return embs
        
        # All embeddings for a conjunction have the same order of times.
        # This can only be assumed in the magic-sequence version, where all possible sequence links are included.
        # Making it a list rather than a generator because a) min complains if you give it a size-0 list and b) it's small anyway.
        times = [ (var,obj) for (var,obj) in embs[0].items() if obj.get_type() == t.TimeNode ]
        if not len(times):
            return embs
        
        def int_from_var_obj(ce):
            return int(ce[1].op.name)
        
        first_time_var_obj = min(times, key=int_from_var_obj)
        first_time, _ = first_time_var_obj
        
        simplified_embs = set()
        for s in embs:
            simple_s = tuple( (var, obj) for (var, obj) in s.items() if obj.get_type() != t.TimeNode or var == first_time  )
            simplified_embs.add(simple_s)
        
        if len(simplified_embs) != len(embs):
            print '++find_exists_embeddings', embs, '=>', simplified_embs
        
        return simplified_embs


def notice_changes(atomspace):    
    tv_delta = 0.000001
    
    t = types
    
    times = atomspace.get_atoms_by_type(t.TimeNode)
    times = [f for f in times if f.name != "0"] # Related to a bug in the Psi Modulator system
    times = sorted(times, key= lambda t: int(t.name) )

    target_PredicateNodes = [x for x in atomspace.get_atoms_by_type(t.PredicateNode) if "DemandGoal" in x.name]

    for atom in target_PredicateNodes:
        target = T('EvaluationLink', [atom])

        # find all of the xDemandGoal AtTimeLinks in order, sort them, then check whether each one is higher/lower than the previous one.       
        
        atTimes = []
        times_with_update = []
        for time in times:
#            # Need to use unify because not sure what the arguments will be. But they must be the same...
#            template = Tree('AtTimeLink', time, target)
#            matches = find_conj( (template,) )
#            
#            # If this DemandGoal is in use there will be one value at each timestamp (otherwise none)
#            assert len(matches) < 2
#            matches[0].
            template = T('AtTimeLink', [time, target])
            a = atom_from_tree(template, atomspace)
            
            # Was the value updated at that timestamp? The PsiDemandUpdaterAgent is not run every cycle so many
            # timestamps will have no value recorded.
            if a.tv.count > 0:
                atTimes.append(a)
                times_with_update.append(time)
    
        if len(atTimes) < 2:
            continue
        
        for i, atTime in enumerate(atTimes[:-1]):
            atTime_next = atTimes[i+1]
            
            tv1 = atTime.tv.mean
            tv2 = atTime_next.tv.mean
            
            print tv2-tv1
            
            if tv2 - tv1 > tv_delta:
                # increased
                pred = 'increased'
            elif tv1 - tv2 > tv_delta:
                # decreased
                pred = 'decreased'
            else:
                continue

            time2 = times_with_update[i+1]

            tv = TruthValue(1, 1.0e35)
            res = T('AtTimeLink',
                     time2,
                     T('EvaluationLink',
                                atomspace.add(t.PredicateNode, name=pred),
                                T('ListLink',
                                    target
                                )
                        )
                    )
            a = atom_from_tree(res, atomspace)
            a.tv = tv
            
            print str(a)
            
            atTime.tv = TruthValue(0, 0)

try:
    class ClockMindAgent(opencog.cogserver.MindAgent):
        def __init__(self):
            self.cycles = 1
    
        def run(self,atomspace):
            times = atomspace.get_atoms_by_type(t.TimeNode)
            times = sorted(times, key= lambda t: int(t.name) )
            
            print times[-1].name

    class FishgramMindAgent(opencog.cogserver.MindAgent):
        def __init__(self):
            self.cycles = 1
    
        def run(self,atomspace):
            # It may be useful to store the fishgram object so you can reuse results in each cycle
            try:
                self.fish
            except:            
                self.fish = Fishgram(atomspace)
                
                #make_seq(atomspace)
                # Using the magic evaluator now. But add a dummy link so that the extractForest will include this
                #atomspace.add(t.SequentialAndLink, out=[atomspace.add(t.TimeNode, '0'), atomspace.add(t.TimeNode, '1')], tv=TruthValue(1, 1))
                
                # Detect timestamps where a DemandGoal got satisfied or frustrated
                notice_changes(atomspace)
    
                self.fish.forest.extractForest()
                print (self.fish.forest.all_trees)
    
            
            #fish.iterated_implications()
            self.fish.implications()
            #self.fish.run()
            print "Finished one Fishgram cycle"
            
            #fish.make_all_psi_rules()
    
            self.cycles+=1

except NameError:
    pass
