try:
    from opencog.atomspace import AtomSpace, Atom, get_type, types
except ImportError:
    from atomspace_remote import AtomSpace, Atom, get_type, types

from copy import copy, deepcopy
from functools import *
import sys
from itertools import permutations
from util import *

from collections import namedtuple
#from rules import Rule
#import rules

def coerce_tree(x):
    '''transform x to a tree with T() '''
    assert type(x) != type(None)
    if isinstance(x, Tree):
        return x
    else:
        return T(x)

##
# @brief   construct a tree, no recursive, Transparently record Links as strings rather than Handles
#
#
# @param op : if it's a atom, would be change to string,  should not be a nonempty link Atom
# @param args
#
# @return 
def T(op, *args):
    if len(args) and isinstance(args[0], list):
        args = args[0]
    # Transparently record Links as strings rather than Handles
    assert type(op) != type(None)
    if len(args):
    # inner node
        if isinstance(op, Atom):
            assert not op.is_a(types.Link)
            final_op = op.type_name
        else:
            final_op = op
        final_args = [coerce_tree(x) for x in args]
    else:
    # leaf
        final_op = op
        final_args = []

    return Tree(final_op, final_args)

def Var(op):
    return Tree(op)

class Tree (object):
    ''' a general tree, have no tranformation of data '''
    def __init__(self, op, args = None):
        # Transparently record Links as strings rather than Handles
        assert type(op) != type(None)
        ## the value of the node, could be int, Atom, or string
        ## ,a link would be converted to string
        self.op = op  
        if args is None:
            ## could be subtree or just immediate child
            self.args = []
        else:
            self.args = args               
        
        self._tuple = None

    def __str__(self):
        if self.is_leaf():
            if isinstance(self.op, Atom):
                #return self.op.name+':'+self.op.type_name
                return self.op.name
            else:
                return '$'+str(self.op)
        else:
            return '(' + str(self.op) + ' '+ ' '.join(map(str, self.args)) + ')'

    # TODO add doctests
    def __repr__(self):
        return str(self)
        if self.is_variable():
            # e.g. T(123)
            return "T(%s)" % (str(int(self.op)),)
        elif self.is_leaf() and isinstance(self.op, Atom):
            # a.add(t.ConceptNode, name='cat')
            return "a.add(t.%s, '%s')" % (self.op.type_name, self.op.name)
        else:
            # T('ListLink', T(a.add(t.ConceptNode, name='eat')) )
            return "T('%s', %s)" % (self.op, repr(self.args))

    def __hash__(self):
        return hash( self.to_tuple() )

    def is_variable(self):
        "A variable is an int starting from 0"
        #return isinstance(self.op, int)
        try:
            self._is_variable
        except:            
            self._is_variable = isinstance(self.op, int)
        return self._is_variable
    
    def get_type(self):
        if self.is_variable():
            return types.Atom
        elif isinstance(self.op, Atom):
            return self.op.t
        else:
            return get_type(self.op)
    
    def is_leaf(self):
        return len(self.args) == 0
    
    def __cmp__(self, other):
        if not isinstance(other, Tree):
            return cmp(Tree, type(other))
        #print self.to_tuple(), other.to_tuple()
        return cmp(self.to_tuple(), other.to_tuple())
    
    def to_tuple(self):
        # Simply cache the tuple.
        # TODO: A more efficient alternative would be to adapt the hash function and compare function
        # to work on Trees directly.
        if self._tuple != None:
            return self._tuple
        else:
            # Atom doesn't support comparing to different types in the Python-standard way.
            if isinstance(self.op, Atom): # and not isinstance(self.op, FakeAtom):
                #assert type(self.op.h) != type(None)
                self._tuple = self.op.h.value()
                return self._tuple
                #return self.op.type_name+':'+self.op.name # Easier to understand, though a bit less efficient
            else:
                self._tuple = tuple([self.op]+[x.to_tuple() for x in self.args])
                return self._tuple

    def isomorphic(self, other):
        return isomorphic_conjunctions_ordered((self, ), (other, ))
    
    def unifies(self, other):
        assert isinstance(other, Tree)
        return unify(self, other, {}) != None

    def canonical(self):
        return canonical_trees([self])[0]

    def flatten(self):
        # @@? could be used to reproduce the tree ?
        # t=Tree('EvaluationLink',Tree(1),Tree('ListLink',Tree('cat'),Tree('dog')))
        return [self]+concat_lists(map(Tree.flatten, self.args))

class DAG(Tree):
    '''double directed Graph '''
    def __init__(self,op,args):
        Tree.__init__(self,op,[])
        self.parents = []
        self.trace = Data_Trace()
        self.tv = TruthValue(0,0)
        for a in args:
            self.append(a)
        try:
           self.trace.path_pre = op.trace.path_pre 
           self.trace.path_axiom = op.trace.path_axiom 
        except Exception:
            pass
    
    def append(self,child):
        if self not in child.parents:
            child.parents.append(self)
            ## children
            self.args.append(child)
    
    def __eq__(self,other):
        if type(self) != type(other):
            return False
        return self.op == other.op

    def __hash__(self):
        return hash(self.op)
    
    def __str__(self):
        return str(self.op)
    
    def any_path_up_contains(self,targets):
        if self in targets:
            return True
        return any(p.any_path_up_contains(targets) for p in self.parents)
        
##
# @brief recursive construct a tree from atom
#
# @param atom
# @param dic : help to replace VariableNode in the tree to consistent int var
#
# @return 
def tree_from_atom(atom, dic = {}):
    if atom.is_node():
        if atom.t in [types.VariableNode, types.FWVariableNode]:
            try:
                return dic[atom]
            except:
                var = new_var()
                dic[atom] = new_var()
                return var
        else:
            return Tree(atom)
    else:
        args = [tree_from_atom(x, dic) for x in atom.out]
        return Tree(atom.type_name, args)

def atom_from_tree(tree, a):
    ''' also add the the tree to atomspace! '''
    if tree.is_variable():
        return a.add(types.VariableNode, name='$'+str(tree.op))
    elif tree.is_leaf():
        # Node (simply the handle)        
        if isinstance (tree.op, Atom):
            return tree.op
        # Empty Link
        else:
            return a.add(get_type(tree.op), out = [])
    else:
        out = [atom_from_tree(x, a) for x in tree.args]
        return a.add(get_type(tree.op), out=out)

def find(template, atoms):
    return [a for a in atoms if unify(tree_from_atom(a), template, {}) != None]

def find_tree(template, atoms):
    def convert(x):
        if isinstance(x,Atom):
            return tree_from_atom(x)
        else:
            return x
    
    return [convert(a) for a in atoms if unify(convert(a), template, {}) != None]

class Match(object):
    def __init__(self, subst = {}, atoms = [], conj = ()):
        self.subst = subst
        self.atoms = atoms
        self.conj = conj
    
    def __eq__(self, other):
        return self.subst == other.subst and self.atoms == other.atoms and self.conj == other.conj

def find_conj(conj, atom_provider, match = Match()):
    """Find all combinations of Atoms matching the given conjunction.
    atom_provider can be either an AtomSpace or a list of Atoms.
    Returns a list of (unique) Match objects."""
    if conj == ():
        return [match]
    
    tr = conj[0]
    
    if isinstance(atom_provider, AtomSpace):
        root_type = tr.get_type()
        atoms = atomspace.get_atoms_by_type(root_type)
    else:
        atoms = atom_provider
    
    ret = []
    for a in atoms:
        s2 = unify(tr, tree_from_atom(a), match.subst)
        if s2 != None:
            match2 = Match(s2, match.atoms+[a])
            
            #print pp(match2.subst), pp(match2.atoms)
            
            later = find_conj(conj[1:], atoms, match2)
            
            for final_match in later:
                if final_match not in ret:
                    ret.append(final_match)
    return ret

def find_matching_conjunctions(conj, trees, match = Match()):
    if conj == ():
        #return [match]
        partially_bound_conj = subst_conjunction(match.subst, match.conj)
        m2 = Match(conj = partially_bound_conj, subst = match.subst)
        return [m2]
    
    ret = []
    for tr in trees:
        tr = standardize_apart(tr)
        s2 = unify(conj[0], tr, match.subst)
        if s2 != None:
            match2 = Match(conj=match.conj+(conj[0],), subst=s2)
            
            #print pp(match2.conj), pp(match2.subst)
            
            later = find_matching_conjunctions(conj[1:], trees, match2)
            
            for final_match in later:
                if final_match not in ret:
                    ret.append(final_match)
    return ret

def apply_rule(precedent, conclusion, atoms):
    ret = []
    for x in atoms:
        if isinstance(x, Atom):
            x = tree_from_atom(x)
        s = unify(precedent, x, {})
        if s != None:
            ret.append( subst(s, conclusion) )
    return ret

# Further code adapted from AIMA-Python under the MIT License (see http://code.google.com/p/aima-python/)
# @@?

##
# @brief unify x to y, return a unification change x(r.head) to y(target)
#
# @param x : a tree
# @param y : a tree
# @param s : dict, source to target map
#
# @return : {}:  match without substitution, None: failed to unify
def unify(x, y, s):
    """Unify expressions x,y with substitution s; return a substitution that
    would make x,y equal, or None if x,y can not unify. x and y can be
    variables (e.g. 1, Nodes, or tuples of the form ('link type name', arg0, arg1...)
    where arg0 and arg1 are any of the above. NOTE: If you unify two Links, they 
    must both be in tuple-tree format, and their variables must be standardized apart.
    >>> ppsubst(unify(x + y, y + C, {}))
    {x: y, y: C}
    """
    #print "unify %s %s" % (str(x), str(y))

    tx = type(x)
    ty = type(y)

    assert not tx == tuple and not ty == tuple

    if s == None:
        return None
    # Not compatible with RPyC as it will make one of them 'netref t'
    elif tx != ty:
        return None
    # Obviously this is redundant with the equality check at the end,
    # but I moved that to the end because it's expensive to check
    # whether two trees are equal (and it has to check it separately for
    # every subtree, even if it's going to recurse anyway)
    #elif x == y:
    #   return s
    elif tx == Tree and ty == Tree and x.is_variable() and y.is_variable() and x == y:
        return s
    # extend s if possible
    elif tx == Tree and x.is_variable():
        return unify_var(x, y, s)
    elif ty == Tree and y.is_variable():
        return unify_var(y, x, s)
    # end extend
        
    elif tx == Tree and ty == Tree:
        # @@?
        # none variable, could be a link or a concept
        s2 = unify(x.op, y.op, s)
        return unify(x.args,  y.args, s2)
    # Recursion to handle arguments.
    elif tx == list and ty == list:
        if len(x) == len(y):
            if len(x) == 0:
                return s
            else:
                # unify all the arguments
                s2 = unify(x[0], y[0], s)
                return unify(x[1:], y[1:], s2)

    elif x == y:
        return s
        
    else:
        return None

def unify_var(var, x, s):
    if var in s:
        #@@? just return s
        return unify(s[var], x, s)
    # check for the first time
    elif occur_check(var, x, s):
        raise ValueError('cycle in variable bindings')
        return None
    else:
        return extend(s, var, x)

def occur_check(var, x, s):
    """Return true if variable var occurs anywhere in x
    (or in subst(s, x), if s has a binding for x)."""

    if x.is_variable() and var == x:
        return True
    elif x.is_variable() and s.has_key(x):
        return occur_check(var, s[x], s)
    # What else might x be? 
    elif not x.is_leaf():
        # Compare link type and arguments
#        return (occur_check(var, x.op, s) or # Not sure that's necessary
#                occur_check(var, x.args, s))
        return any([occur_check(var, a, s) for a in x.args])
    else:
        return False

##
# @brief add the map between 'var' and 'val'
#
# @param s
# @param var
# @param val
#
# @return 
def extend(s, var, val):
    """Copy the substitution s and extend it by setting var to val;
    return copy.
    
    >>> initial = {x: a}
    >>> extend({x: a}, y, b)
    {y: b, x: a}
    """
    s2 = s.copy()
    s2[var] = val
    return s2
def subst(s, x):
    """Substitute the substitution s into the expression x.
    >>> subst({x: 42, y:0}, F(x) + y)
    (F(42) + 0)
    """
#    if isinstance(x, Atom):
#        return x
#    elif x.is_variable(): 
#        # Notice recursive substitutions. i.e. $1->$2, $2->$3
#        # This recursion should also work for e.g. $1->foo($2), $2->bar
#        return subst(s, s.get(x, x))
    if x.is_variable():
        return s.get(x, x)
    elif x.is_leaf(): 
        return x
    else: 
        #return tuple([x[0]]+ [subst(s, arg) for arg in x[1:]])
        return Tree(x.op, [subst(s, arg) for arg in x.args])

def subst_conjunction(substitution, conjunction):
    ret = []
    for tr in conjunction:
        ret.append(subst(substitution, tr))
    return tuple(ret)

def subst_from_binding(binding):
    return dict([ (Tree(i), obj) for i, obj in enumerate(binding)])

def binding_from_subst(subst, atomspace):
    #return [ atom_from_tree(obj_tree, atomspace) for (var, obj_tree) in sorted(subst.items()) ]
    return [ obj_tree for (var, obj_tree) in sorted(subst.items()) ]

def bind_conj(conj, b):
    return subst_conjunction(subst_from_binding(b), conj)

##
# @brief :replace all the variables in tree with new variables
#
# @param tr : a tree of a tuple of tree or a tuple of tuple of tree
# @param dic :the map between the tree and new variable
#
# @return :a tree or a tuple
def standardize_apart(tr, dic=None):
    if dic == None:
        dic = {}

    if isinstance(tr, tuple):
        return tuple([standardize_apart(a, dic) for a in tr])
    elif tr.is_variable():
        if tr in dic:
            return dic[tr]
        else:
            v = new_var()
            dic[tr] = v
            return v
    else:
        return Tree(tr.op, [standardize_apart(a, dic) for a in tr.args])


def new_var():
    '''return a new variable tree node '''
    global _new_var_counter
    _new_var_counter += 1
    return Tree(_new_var_counter)
## the begin of new variables
_new_var_counter = 10**6

def unify_conj(xs, ys, s):
    '''unify conjunction of trees '''
    assert isinstance(xs, tuple) and isinstance(ys, tuple)
    if len(xs) == len(ys):
        for perm in permutations(xs):
            # iterate over N x N possible unifies, util find one
            s2 = unify(list(perm), list(ys), s)
            if s2 != None:
                return s2
    else:
        return None

def isomorphic_conjunctions(xs, ys):
    if isinstance(xs, tuple) and isinstance(ys, tuple) and len(xs) == len(ys):
        for perm in permutations(xs):
            if isomorphic_conjunctions_ordered(tuple(perm), ys):
                return True
    return False

def isomorphic_conjunctions_ordered(xs, ys):
    ''' return true if just variables'name of two tree are different''' 
    xs, ys = canonical_trees(xs), canonical_trees(ys)
    return xs == ys


##
# @brief 
#
# @param trs
# @param dic : same as function standardize_apart
#
# @return : a list of canonical trees
def canonical_trees(trs, dic = {}):
    '''Returns the canonical version of a list of trees, i.e. with the variables renamed (consistently) from 0,1,2.
    It can be a list of only one tree. If you want to use multiple trees, you must put them in the same list, so that
    any shared variables will be renamed consistently.'''

    global _new_var_counter
    tmp = _new_var_counter
    # make variable number begin from 0
    _new_var_counter = 0
    ret = []
    dic = {}
    for tr in trs:
        tr2 = standardize_apart(tr, dic)
        ret.append(tr2)
    _new_var_counter = tmp
    
    return ret

##
# @brief    
#
# @param t : tuple of conjunction trees, or Tree
#
# @return : a list of variables in tree
def get_varlist(t):
    """Return a list of variables in tree, in the order they appear (with depth-first traversal). Would also work on a conjunction."""
    if isinstance(t, Tree) and t.is_variable():
        return [t]
    elif isinstance(t, Tree):
        ret = []
        for arg in t.args:
            ret+=([x for x in get_varlist(arg) if x not in ret])
        return ret
    # Representing a conjunction as a tuple of trees.
    elif isinstance(t, tuple):
        ret = []
        for arg in t:
            ret+=([x for x in get_varlist(arg) if x not in ret])
        return ret
    else:
        return []


def ppsubst(s):
    """Print substitution s"""
    ppdict(s)

class Data_Trace(object):
    """docstring for Data_Trace"""
    def __init__(self,):
        # when tv > 0, is fact!
        self.is_fact = False
        self.path_pre = None
        self.path_axiom = None
        #self.tv = TruthValue(0,0)
