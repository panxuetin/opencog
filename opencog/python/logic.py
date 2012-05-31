#import pyximport; pyximport.install()

#from IPython.Debugger import Tracer; debug_here = Tracer()

try:
    from opencog.atomspace import AtomSpace, types, Atom, TruthValue, get_type_name
    import opencog.cogserver
except ImportError:
    from atomspace_remote import AtomSpace, types, Atom, TruthValue, get_type_name
from tree import *
from util import pp, OrderedSet, concat_lists, inplace_set_attributes
from pprint import pprint
import pdb
from  inference_analyze import *
#try:
#    from opencog.util import log
#except ImportError:
from util import log

log.setLevel(log.DEBUG)
log.setLevel(log.INFO)
log.setLevel(log.WARNING)
log.use_stdout(True)
log.trace = True
from collections import defaultdict

import formulas
import rules

from sys import stdout
# You can use kcachegrind on cachegrind.out.profilestats
#from profilestats import profile

#import line_profiler
#profiler = line_profiler.LineProfiler()

from time import time
import exceptions
#inspect.currentframe().f_lineno
#_line = 1

class Chainer:
    # Convert Atoms into FakeAtoms for Pypy/Pickle/Multiprocessing compatibility
    _convert_atoms = False

    def __init__(self, space, planning_mode = False):
        self.deduction_types = ['SubsetLink', 'ImplicationLink', 'InheritanceLink']

        ## [app1.goals, app1.head, app2.goals, ...]
        self.bc_later = OrderedSet()
        ## help to deside whether a target has been  expanded before
        self.bc_before = OrderedSet()

        self.fc_later = OrderedSet()
        self.fc_before = OrderedSet()
        self.pd = dict()
        self.is_result = False

        self.results = []
        self.progating = False
        ## the beigin app that progating up (is the buttom inner node of the tree)
        self.begin_app = None
        # flag help to locate the begin app that progating up
        self.begin_progating = False
        self.valid_begin = []

        self.space = space
        self.planning_mode = planning_mode
        self.viz = PLNviz(space)
        self.viz.connect()
        self.setup_rules()
        self.bc_step_count = 0
        
        global _line
        _line = 1
        
        #profiler.add_function(self.bc)
    
    # @brief        
    #
    # @param target :a tree
    #
    # @return 
    def bc(self, target):
        '''
        target : a tree
        '''
        #import prof3d; prof3d.profile_me()
        
        #try:
        #print "target:" 
        #print target
        
        self.results = []

        self.target = target
        # Have a sentinel application at the top
        self.root = T('WIN')
        self.rules.append(rules.Rule(self.root,[target],name='producing target'))
        # the begining of expanding node
        self.bc_later = OrderedSet([self.root])
        # viz - visualize the root
        #self.viz.outputTarget(target, None, 0, 'TARGET')

        while self.bc_later and not self.results:
        #while self.bc_later:
            self.bc_step()
        # Always print it at the end, so you can easily check the (combinatorial) efficiency of all tests after a change
        log.debug(format_log(1,False,"bc_step:" + str(self.bc_step_count)))
        log.debug("------------------bc_before :-------------------------")
        for expand in self.bc_before:
            log.debug(format_log(1,False, str(expand)))
        log.debug("------------------results :-------------------------")
        for expand in self.results:
            log.debug(format_log(1,False, str(expand)))

    def bc_step(self):
        assert self.bc_later
        self.bc_step_count += 1
        # @@! standardize_apart before insert to bc_before
        #next_target = self.get_fittest(self.bc_later) # Best-first search
        next_target = self.bc_later.pop_last()
        ##@@@
        #next_target = standardize_apart(next_target)
        self.add_tree_to_index(next_target, self.bc_before)
        # app rules could reach target
        log.debug(format_log(0,False, "\n"))
        log.debug(format_log(0,True,"***************************** expanding: %s ********************************" % str(next_target)))
        apps = self.find_rule_applications(next_target)   # @@% n of number of rules
        for a in apps:                     # @@% n of valid rules
            a = a.standardize_apart()
            # @1 add to bc_later and search tree
            if self.add_queries(a):
                log.debug(format_log(4,True,"adding app: %s" % str(a)))
                # prove goals with axioms
                self.find_axioms_for_rule_app(a)

        return

    def contains_isomorphic_tree(self, tr, idx):        
        #return any(expr.isomorphic(tr) for expr in idx)
        canonical = tr.canonical()
        return canonical in idx

    def add_tree_to_index(self, tr, idx):
        ''' canonical the tr(tree) and add to idx '''
        # idx = self.bc_later
        # @@@
        canonical = tr.canonical()
        #canonical = standardize_apart(tr)
        idx.append(canonical)

    def _app_is_stupid(self, goal):

        # You should probably skip the app entirely if it has any self-implying goals
        def self_implication(goal):
            return any(goal.get_type() == get_type(type_name) and len(goal.args) == 2 and goal.args[0].isomorphic(goal.args[1])
                        for type_name in self.deduction_types)

        # Nested ImplicationLinks
        # skip Implications between InheritanceLinks etc as well
        types = map(get_type, self.deduction_types)
        if (goal.get_type() in types and len(goal.args) == 2 and
                (goal.args[0].get_type() in types or
                 goal.args[1].get_type() in types) ):
            return True

        try:
            self._very_vague_link_templates
        except:
            self._very_vague_link_templates = [standardize_apart(T(type, 1, 2)) for type in self.deduction_types]

        # Should actually block this one if it occurs anywhere, not just at the root of the tree.
        #very_vague = any(goal.isomorphic(standardize_apart(T(type, 1, 2))) for type in self.deduction_types)
        very_vague = any(goal.isomorphic(template) for template in self._very_vague_link_templates)
        return (self_implication(goal) or
                     very_vague)

    def reject_expression(self,expr):
        return self._app_is_stupid(expr) or expr.is_variable()

    def add_app_dags(self, app):
        """docstring for add_to_dags"""
        if any(map(self._app_is_stupid, app.goals)) or self._app_is_stupid(app.head):
            #log.debug(format_log(1, True, "stupid app:" + str(app)))
            return False
        # If the app is a cycle or already added, don't add it or any of its goals
        # inference tree
        (status, app_pdn) = self.add_app_to_pd(app)
        #if status == 'NEW':
            ## Only visualize it if it is actually new
            ## viz
            #for (i, input) in enumerate(app.goals):
                #self.viz.outputTarget(input.canonical(), app.head.canonical(), i, app)

        if status == 'CYCLE' or status == 'EXISTING':
            return False
        else:
            # add a new app to dags
            return True
            
    ##
    # @brief :Key function add app to self.bc_later and search tree
    #
    # @param app
    #
    # @return 
    def add_queries(self, app):
        ## @todo seperate bc_later and search path
        ## @todo Rule_Trace Inheritance from Rule
        ## @todo why no a -> c
        def goal_is_stupid(goal):
            if goal.is_variable():
                #log.info(format_log(1, "*** stupid goal:%s in app: %s"%(str(goal),str(app)),inspect.currentframe().f_lineno))
                return True
            else:
                return False

        if any(map(self._app_is_stupid, app.goals)) or self._app_is_stupid(app.head):
            #log.debug(format_log(1, True, "stupid app:" + str(app)))
            return 
        # If the app is a cycle or already added, don't add it or any of its goals
        # inference tree
        (status, app_pdn) = self.add_app_to_pd(app)
        #if status == 'NEW':
            ## Only visualize it if it is actually new
            ## viz
            #for (i, input) in enumerate(app.goals):
                #self.viz.outputTarget(input.canonical(), app.head.canonical(), i, app)
        if status == 'CYCLE' or status == 'EXISTING':
            return 
        # add goals to self.pd
        # NOTE: For generators, the app_pdn will exist already for some reason
        # It's useful to add the head if (and only if) it is actually more specific than anything currently in the BC tree.
        # This happens all the time when atoms are found.
        all_stupid = True
        for goal in tuple(app.goals):      # @@% n the number of goals 
            if not goal_is_stupid(goal):
                if  (not self.contains_isomorphic_tree(goal, self.bc_before) and   # @@% the length of (bc_before + bc_later)
                     not self.contains_isomorphic_tree(goal, self.bc_later) ):
                    assert goal not in self.bc_before
                    assert goal not in self.bc_later
                    self.add_tree_to_index(goal, self.bc_later)
                    all_stupid = False
                    log.debug(format_log(log.ident + 6,True, "^^adding goal: %s "%(str(goal))))
        #log.debug(str(self.bc_later))
        if not all_stupid:  
            return app

    def check_premises(self, app):
        '''Check whether the given app can produce a result. This will happen if all its premises are
        already proven. Or if it is one of the axioms given to PLN initially. It will only find premises
        that are exactly isomorphic to those in the app (i.e. no more specific or general). The chainer
        itself is responsible for finding specific enough apps.'''
        input_tvs = [self.get_tvs(goal) for goal in app.goals]   # @@% n(the number of goals) * args(the number of child in each goal)
        res = all(tvs != [] for tvs in input_tvs)
        #print "***check: " + str(app)
        #print res
        return res
    
    def compute_and_add_tv(self, app):
        # NOTE: assumes this is the real copy of the rule, not just a new one.
        #app.tv = True 
        input_tvs = [self.get_tvs(g) for g in app.goals]   # @@% n(the number of goals) * args(the number of child in each goal)
        assert all(input_tvs)
        input_tvs = [tvs[0] for tvs in input_tvs]
        input_tvs = [(tv.mean, tv.count) for tv in input_tvs]
        tv_tuple = app.formula(input_tvs,  None)
        app.tv = TruthValue(tv_tuple[0], tv_tuple[1])
        assert app.tv.count > 0
        self.set_tv(app, app.tv)
            

    def find_rule_applications(self, target):
        ''' rules could reach the target '''
        ret = []
        for r in self.rules:
            if not r.goals:
                # "WIN" <- target, make sure it work all right
                continue
            assert r.match == None
            if target == self.root and r.name != 'producing target':
                # @@? 
                # self.root = "WIN" , make sure the path go through "WIN" <- target, so the target is proved
                continue
            s = unify(r.head, target, {})
            if s != None:
                log.debug(format_log(1,True,"valid rule: %s" % str(r)))
                new_rule = r.subst(s)
                new_rule.trace.path_pre = r
                new_rule.trace.path_axiom = target
                ret.append(new_rule)
        return ret

    def find_axioms_for_rule_app(self, app):
        ''' app: standardize_aparted substitued rule could possiblly reach the target "a->c| a->x, x->c, a, x, c" 
            rule always have head and goals
        '''
        ##
        # @brief : the entrance to propagate
        #
        # @param axiom_app :standardize_aparted rule of axiom could be unified to specific goal, like a -> b
        # @param s :substitution of goal a->x to axiom a->b 
        #
        # @return 
        def found_axiom(axiom_app, s):
            if axiom_app != None:
                # add "a->b" to search path
                # @@! @2 add the axiom to searth path
                log.debug(format_log(10,True,"[axiom]: %s" % str(axiom_app)))
                self.add_app_to_pd(axiom_app)
                log.ident += 10
                if len(s) == 0:
                    # goal match without substitution
                    assert goal == axiom_app.head
                    self.begin_progating = True
                    self.propagate_result(app)
                else:
                    # goal math with substitution
                    self.propagate_specialization(axiom_app.head, i, app)
                log.ident = 0
        # to prove goals 
        for (i, goal) in enumerate(app.goals):   # @@% n, the number of goal
            # @@!
            if self.reject_expression(goal): # goal is a variable
                continue
            #raw_input("next step:") 
            log.debug(format_log(7,True,"with goal: %s" % str(goal)))
            for r in self.rules:           # @@% n, the number of self.rules
                # iterate over axioms
                axiom_app = None
                if len(r.goals):
                    continue
                # r.head == axiom, could be rule like fact
                r = r.standardize_apart()
                if r.match == None:
                    s = unify(r.head, goal, {})   # @@% n, unification
                    if s != None:
                        # a rule of axiom, not a tree of axiom
                        axiom_app = r.subst(s)   # @@% n, the length of r
                        found_axiom(axiom_app, s)
                else:
                    assert False

    def propagate_result(self, app):
        '''
           app: a more specialized and standardize_aparted app rule, or an app whose head match an axiom perfect
        '''
        # when invoked by "found_axiom, means": one goal math perfect, it check if all sibling goal reachable
        # when invoked by "propagate_specialization", means: an more specialized premise, checking if the premise is true
        # actually, it's always false if the one of the premise's goal contain variable 
        got_result = self.check_premises(app)    # @@% n(the number of goals) * args(the number of child in each goal)
        #log.info(format_log(1, "****check premise:%s"%str(app),inspect.currentframe().f_lineno))
        self.got_target = False
        log.ident += 3
        if got_result:
            if self.begin_progating:
                self.begin_app = app
                self.begin_progating = False
            self.progating = True
            log.debug(format_log(log.ident,True, "##sucess! " + str(app)))
            #  have added by propagate_specialization or bc_step already!
            self.compute_and_add_tv(app)      # @@% n(the number of goals) * args(the number of child in each goal)
            # mark as fact
            #a = self.expr2pdn(app)
            #a.is_fact = True
            if app.head == self.root:
                # could search mutiple different results at the same time @@!
                self.results.append(app.goals[0])
                self.valid_begin.append(self.begin_app)
                self.got_target = True
                return
            # then propagate the result further upward
            # add  the fact inferenced !
            result_pdn = self.expr2pdn(app.head.canonical())
            # @@! propagate up
            upper_pdns = [app_pdn for app_pdn in result_pdn.parents]
            #assert(len(upper_pdns)==0)
            for app_pdn in upper_pdns:
                self.progating = True
                self.propagate_result(app_pdn.op)   # @@% n(thie number of parents) * progating_result()
            log.ident -= 3
            return True
        else:
            self.progating = False
            log.debug(format_log(log.ident+3,True, "##failed this time"))
            log.ident -= 3

            return False
    
    ##
    # @brief to propagate a premise
    #
    # @param axiom_goal : an more specialized node(could be an axiom) result from unification of the ith goal of orig_app, like "a->b" 
    # @param i : the index of goal in an substitued rule
    # @param orig_app :the standardize_aparted and substitued rule (premise), like " a->c | a->x, x->c, a, x, c" 
    #
    # @return 
    def propagate_specialization(self, axiom_goal, i, orig_app):
        axiom_goal = standardize_apart(axiom_goal)
        s = unify(orig_app.goals[i], axiom_goal, {})   # @@% n, unify
        assert orig_app.trace.path_pre
        assert s != None and s != { }
        orig_app_pdn = self.app2pdn(orig_app)
        assert len(orig_app_pdn.parents) == 1
        orig_result_pdn = orig_app_pdn.parents[0]   # "a -> c" 
        # a more specialized app rule, could be true or false
        new_app = orig_app.subst(s)
        new_app.trace.path_pre = orig_app
        new_app.trace.path_axiom = axiom_goal
        # @3
        if not self.add_queries(new_app):
            #log.debug(format_log(log.ident,True, "**existing or cycle app: " + str(new_app) ))
            return
        self.begin_progating = True
        log.ident += 3
        log.debug(format_log(log.ident,True, "##premise: " + str(new_app)))
        # @@! when propagate_result reach the target, then it get a more general answer, more specialization is useless, because it can only get more specialized answer
        # determine if the premise is valuable
        self.propagate_result(new_app)     # @@% propagate_result
        log.ident -= 3
        # probably, i may never happen that progate_result to target, and new_app.head != orig_app.head
        # because this would imply that goal like "a -> $1" may be an infernced fact or axiom
        # which would unlikely happen, because $1 is just for rule, not variables in fact
        # so check self.got_target is unncessary
        # Specialize higher-level rules, if the head has actually been changed 
        # (if the unify lessen the number of vars in the head of orig_app)
        # which imply the head include the same var as current goal
        if new_app.head != orig_app.head:
            # more specialized head 
            new_result = new_app.head          # "a -> c" 
            # when the result of orig_app still include var after unified, it progate to the parent of orig_app
            for higher_app_pdn in orig_result_pdn.parents:   # @@% n(the number of parents) * propagate_specialization
                j = higher_app_pdn.args.index(orig_result_pdn)
                self.propagate_specialization(new_result, j, higher_app_pdn.op)
        # begin prove more specialized goals of new_app in next step_bc

    # used by forward chaining
    def find_new_rule_applications_by_premise(self,premise):
        ret = []
        for r in self.rules:
            for g in r.goals:
                s = unify(g, premise, {})
                if s != None:
                    new_rule = r.subst(s)
                    ret.append(new_rule)
        return ret

    def get_tvs(self, expr):
        ''' return true values of all goal '''
        # Only want to find results for this exact target, not every possible specialized version of it.
        # The chaining mechanism itself will create different apps for different specialized versions.
        # If there are any variables in the target, and a TV is found, that means it has been proven
        # for all values of that variable.
        canonical = expr.canonical()
        expr_pdn = self.expr2pdn(canonical)
        # when is axiom args is the axiom rule, expr is the head of axiom rule
        # @@! args, not goals
        app_pdns = expr_pdn.args
        tvs = [app_pdn.tv for app_pdn in app_pdns if app_pdn.tv.count > 0]
        return tvs
    
    def expr2pdn(self, expr):
        '''
            expr: expr of type tree
            return : a node of DAG, with "self.op" == expr, and args = []
        '''
        pdn = DAG(expr,[])
        try:
            return self.pd[pdn]
        except KeyError:
            #print 'expr2pdn adding %s for the first time' % (pdn,)
            self.pd[pdn] = pdn
            return pdn

    def set_tv(self,app,tv):
        assert isinstance(app, rules.Rule)
        # Find/add the app
        # get a existing app in search tree
        a = self.app2pdn(app)
        assert not a is None
        a.tv = tv
    
    ##
    # @brief :Key function that record the search path 
    #
    # @param app :substitued rule
    #
    # @return :  a node of DAG, it's parent is app.head, it's children are goals, and itself is DAG(app)
    def add_app_to_pd(self,app):
        def canonical_app_goals(goals):
            return map(Tree.canonical, goals)
        goals_canonical = canonical_app_goals(app.goals)
        # Don't allow loops. Does this need to be a separate test? It should probably
        # check whether the new target is more specific, not just equal?
        # not standardize_apart goals, insure the unique
        goal_pdns = [self.expr2pdn(g) for g in goals_canonical]
        head_pdn = self.expr2pdn(app.head.canonical())
        # @@!
        if head_pdn.any_path_up_contains(goal_pdns):
            return ('CYCLE',None)
        # Check if this application is in the Proof DAG already.
        # NOTE: You must use the app's goals rather than the the app PDN's arguments,
        # because the app's goals may share variables.
        # @@!
        # @@% n(number of sibling) * m(number of goals of sibling)
        existing = [apn for apn in head_pdn.args if canonical_app_goals(apn.op.goals) == goals_canonical]
        assert len(existing) < 2
        if len(existing) == 1:
            return ('EXISTING',existing[0])
        else:
            # Otherwise add it to the Proof DAG
            app_pdn = DAG(app,[])
            # @@!
            app_pdn.tv = app.tv
            for goal_pdn in goal_pdns:
                app_pdn.append(goal_pdn)
            head_pdn.append(app_pdn)
            return ('NEW',app_pdn)

    def app2pdn(self,app):
        (status,app_pdn) = self.add_app_to_pd(app)
        assert status != 'CYCLE'
        return app_pdn
    
    def add_rule(self, rule):
        self.rules.append(rule)
        
        # This is necessary so get_tvs will work
        # Only relevant to generators or axioms
        if rule.tv.confidence > 0:
            self.app2pdn(rule)

    def setup_rules(self):
        self.rules = []
        for r in rules.rules(self.space, self.deduction_types):
            self.add_rule(r)

    def extract_plan(self, trail):
        # The list of actions in an ImplicationLink. Sometimes there are none,
        # sometimes one; if there is a SequentialAndLink then it can be more than one.
        def actions(proofnode):            
            target = proofnode.op
            if isinstance(target, rules.Rule):
                return []
            if target.op in ['ExecutionLink',  'SequentialAndLink']:
                return [pn.op]
            else:
                return []
        # Extract all the targets in best-first order
        proofnodes = trail.flatten()
        proofnodes.reverse()
        actions = concat_lists([actions(pn) for pn in proofnodes])
        return actions

    def trail(self, target):
        def rule_found_result(rule_pdn):
            return rule_pdn.tv.count > 0

        def recurse(rule_pdn):
            #print repr(rule_pdn.op),' PDN args', rule_pdn.args
            exprs = map(filter_expr,rule_pdn.args)
            return DAG(rule_pdn.op, exprs)

        def filter_expr(expr_pdn):
            successful_rules = [recurse(rpdn) for rpdn in expr_pdn.args if rule_found_result(rpdn)]
            return DAG(expr_pdn.op, successful_rules)
        
        root = self.expr2pdn(target)

        return filter_expr(root)
    
    def print_tree(self, tr, level = 1):
        print ' '*(level-1)*3, tr.op
        
        for child in tr.args:
            self.print_tree(child, level+1)


    def bc_score(self, expr):
        def get_coords(expr):
            name = expr.op.name
            tuple_str = name[3:]
            coords = tuple(map(int, tuple_str[1:-1].split(',')))
            return coords

        def get_distance(expr):
            # the location being explored
            l = get_coords(expr)
            # the goal
            t = (90, 90, 98)#get_coords(self.target)
            xdist, ydist, zdist = abs(l[0] - t[0]), abs(l[1] - t[1]), abs(l[2] - t[2])
            dist = xdist+ydist+zdist
            #dist = (xdist**2.0+ydist**2.0+zdist**2.0)**0.5
            return dist
        
        if expr.get_type() == t.ConceptNode and expr.op.name.startswith('at '):
            return -get_distance(expr)
        else:
            return 1000

    def get_fittest(self, queue):
        def get_score(expr):
            expr_pdn = self.expr2pdn(expr)
            
            try:
                return expr_pdn.bc_score
            except AttributeError:
                expr_pdn.bc_score = self.bc_score(expr)
                return expr_pdn.bc_score
        
        # PDNs with the same score will be explored in the order they were added,
        # ie breadth-first search
        best = max(queue, key = get_score)
        queue.remove(best)
        
        return best


from urllib2 import URLError
def check_connected(method):
    '''A nice decorator for use in visualization classes that stream graphs to Gephi. It catches exceptions raised
    when you aren't running Gephi.'''
    def wrapper(self, *args, **kwargs):
        if not self.connected:
            return

        try:
            method(self, *args, **kwargs)
        except URLError:
            self.connected = False

    return wrapper

from collections import defaultdict
import pygephi
class PLNviz:

    def __init__(self, space):
        self._as = space
        self.node_attributes = {'size':10, 'r':0.0, 'g':0.0, 'b':1.0}
        self.rule_attributes = {'size':10, 'r':0.0, 'g':1.0, 'b':1.0}
        self.root_attributes = {'size':20, 'r':1.0, 'g':1.0, 'b':1.0}
        self.result_attributes = {'r':1.0, 'b':0.0, 'g':0.0}

        self.connected = False
        
        self.parents = defaultdict(set)

    def connect(self):
        try:
            self.g = pygephi.JSONClient('http://localhost:8080/workspace0', autoflush=True)
            self.g.clean()
            self.connected = True
        except URLError:
            self.connected = False

    @check_connected
    def outputTarget(self, target, parent, index, rule=None):
        if str(target) == '$1':
            return
        
        self.parents[target].add(parent)

        #target_id = str(hash(target))
        target_id = str(target)

        if parent == None and target != None:
            self.g.add_node(target_id, label=str(target), **self.root_attributes)

        if parent != None:
            # i.e. if it's a generator, which has no goals and only
            # produces a result, which is called 'parent' here
            if target != None:
                self.g.add_node(target_id, label=str(target), **self.node_attributes)

            #parent_id = str(hash(parent))
            #link_id = str(hash(target_id+parent_id))
            parent_id = str(parent)
            #rule_app_id = 'rule '+repr(rule)+parent_id
            rule_app_id = 'rule '+repr(rule)+parent_id
            target_to_rule_id = rule_app_id+target_id
            parent_to_rule_id = rule_app_id+' parent'

            self.g.add_node(rule_app_id, label=str(rule), **self.rule_attributes)

            self.g.add_node(parent_id, label=str(parent), **self.node_attributes)

            # Link parent to rule app
            self.g.add_edge(parent_to_rule_id, rule_app_id, parent_id, directed=True, label='')
            # Link rule app to target
            self.g.add_edge(target_to_rule_id, target_id, rule_app_id, directed=True, label=str(index+1))

    @check_connected
    def declareResult(self, target):
        target_id = str(target)
        self.g.change_node(target_id, **self.result_attributes)

        #self.g.add_node(target_id, label=str(target), **self.result_attributes)

    @check_connected
    # More suited for Fishgram
    def outputTreeNode(self, target, parent, index):
        #target_id = str(hash(target))
        target_id = str(target)

        if parent == None:
            self.g.add_node(target_id, label=str(target), **self.root_attributes)

        if parent != None:
            self.g.add_node(target_id, label=str(target), **self.node_attributes)

            #parent_id = str(hash(parent))
            #link_id = str(hash(target_id+parent_id))
            parent_id = str(parent)
            link_id = str(parent)+str(target)

            self.g.add_node(parent_id, label=str(parent), **self.node_attributes)
            self.g.add_edge(link_id, parent_id, target_id, directed=True, label=str(index))

