from collections import namedtuple, defaultdict

from pprint import pprint

Plan = namedtuple('Plan', ['agenda', 'causal', 'ordering', 'steps'])
def pp_plan(self):
    #return '\nsteps=%s\nagenda=%s\ncausal=%s\nordering=%s\n' % (self.steps,self.causal,self.ordering,self.agenda)
    map(pprint,(self.steps,self.causal,self.ordering,self.agenda))

_next_step_id = 0
def new_step(action):
    global _next_step_id
    
    #step = (action,_next_step_id)
    step = action+'_'+str(_next_step_id)
    _next_step_id += 1
    return step

def get_action_from_step(step):
    (action,id) = step.split('_')
    return action

# record statistics about the combinatorial explostion!
layer_sizes = defaultdict(int)
complete_plans_in_layer = defaultdict(int)

Implication = namedtuple('Implication', ['action','preconditions','postconditions'])
implications = []
action2imp = {}
post2imp = defaultdict(list)

def add_implication(imp):
    implications.append(imp)
    action2imp[imp.action] = imp
    for post in imp.postconditions:
        post2imp[post].append(imp)

add_implication(Implication('unstack(C,A)', frozenset(['clear(C)','on(C,A)','handempty']), frozenset(['holding(C)','clear(A)','~clear(C)','~on(C,A)','~handempty'])))
add_implication(Implication('stack(A,B)', frozenset(['holding(A)','clear(B)']), frozenset(['on(A,B)','clear(A)','handempty','~holding(A)','~clear(B)'])))
add_implication(Implication('stack(B,C)', frozenset(['holding(B)','clear(C)']), frozenset(['on(B,C)','clear(B)','handempty','~holding(B)','~clear(C)'])))
add_implication(Implication('pickup(A)', frozenset(['ontable(A)','clear(A)','handempty']), frozenset(['holding(A)','~ontable(A)','~clear(A)','~handempty'])))
add_implication(Implication('pickup(B)', frozenset(['ontable(B)','clear(B)','handempty']), frozenset(['holding(B)','~ontable(B)','~clear(B)','~handempty'])))
add_implication(Implication('putdown(C)', frozenset(['holding(C)']), frozenset(['ontable(C)','clear(C)','handempty','~holding(C)'])))

start = new_step('start')
end = new_step('end')

def conflict(atom1, atom2):
    spl1 = atom1.split('~')
    spl2 = atom2.split('~')
    
    same_atom = spl1[-1] == spl2[-1]
    same_sign = len(spl1) == len(spl2)
    
    return same_atom and not same_sign

def choose_total_order(plan):
    assert isinstance(plan.ordering, list)
    assert isinstance(plan.steps, list)
    
    def compare(step1, step2):
        s1_before_s2 = step_before(step1, step2, plan.ordering)
        s2_before_s1 = step_before(step2, step1, plan.ordering)
        if s1_before_s2:
            return -1
        elif s2_before_s1:
            return +1
        else:
            return 0

    return sorted(plan.steps, cmp=compare)
    

def is_cyclic(constraint, ordering):
    assert isinstance(ordering, list)
    (before, after) = constraint

    def helper(x):
        if x == before:
            return True
        else:
            return any(helper(y) for y in successors(x, ordering))
    
    return helper(after)

def successors(step, ordering):
    assert isinstance(ordering, list)
    successor_steps = [n for (p,n) in ordering if p == step]
    return successor_steps

def step_before(step1, step2, ordering):
    '''Test if step1 is before step2. It will return False if they are
    the same step, or step1 is after step2, or the ordering doesn't require
    step1 to be before step2. This function will only work if the ordering does
    not contain a cycle (see is_cyclic).'''
    assert isinstance(ordering, list)

    if step1 == step2:
        return False

    def helper(x):
        if x == step2:
            return True
        else:
            return any(helper(y) for y in successors(x, ordering))
    
    return helper(step1)

def pop(initial_state, goal_state):
    
    steps = [start, end]
    ordering = [(start, end)]
    agenda = [(goal,end) for goal in goal_state]
    
    add_implication(Implication('start',[],initial_state))
    add_implication(Implication('end',goal_state,[]))
    
    initial_plan = Plan(steps=steps, ordering = ordering, causal = [], agenda=agenda)
    
    return dfs_solve_precondition(initial_plan, 0)

def dfs_solve_precondition(plan, depth):
    layer_sizes[depth] += 1
    assert isinstance(plan.ordering, list)
    if len(plan.agenda) == 0:
        print "success: ",plan
        complete_plans_in_layer[depth] += 1
        return [plan]

    if depth >= 40:
        return []
    
    (condition, step) = plan.agenda[0]
    agenda = plan.agenda[1:]
    plan = plan._replace(agenda=agenda)
    plans = solve_precondition(plan, condition, step, depth)
    
    return plans

#def dfs_resolve_threats(plan, depth):
#    assert isinstance(plan.ordering, list)
#    if len(plan.agenda) == 0:
#        print "success: ",plan
#        return [plan]
#    
#    resolved_plans = resolve_threats(plan, depth)
#    
#    if len(resolved_plans) == 0:
#        print '   '*depth+'no resolutions'
#    
#    extended_plans = []
#    for p in resolved_plans:
#        extended_plans += dfs_solve_precondition(p, depth+1)
#    
#    return extended_plans

def solve_precondition(plan, goal, step, depth):
    print '   '*depth+'solve_precondition',goal,step
    #pp_plan(plan)
    #print 'S',
    assert isinstance(plan, Plan)
    
    plans = []
    
    # Simple Establishment: Reuse a step already in the plan
    for s in plan.steps:
        if s == step:
            continue
        
        action = get_action_from_step(s)
        imp = action2imp[action]
        post = imp.postconditions
        if goal not in post:
            continue
        
        new_causal_link = (s,goal,step)
        causal = plan.causal+[new_causal_link]

        ordering = plan.ordering[:]
        if is_cyclic((s,step), ordering):
            continue
        if (s,step) not in ordering:
            ordering.append((s,step))
        
        steps = plan.steps[:]
    
        threats = plan_threats(plan._replace(ordering=ordering), s, new_causal_link)
    
        new_plan = plan._replace(causal=causal,ordering=ordering,steps=steps)
        print '   '*depth+"reuse step", s, ":", new_plan.steps
        
        plans += resolve_threats(new_plan, threats, depth+1)

    # Step Addition: Add a new step which achieves goal
    imps = post2imp[goal]
    for imp in imps:
        action = imp.action
        
        if action in ['start','end']:
            continue
        
        assert(goal in imp.postconditions)
        
        s = new_step(action)
        new_causal_link = (s,goal,step)
        causal = plan.causal+[new_causal_link]

        ordering = plan.ordering[:]
        assert not is_cyclic((start,s), ordering)
        ordering.append((start,s))
        assert not is_cyclic((s,end), ordering)
        ordering.append((s,end))
        # If step is end, then this is redundant
        if step != end:
            if is_cyclic((s,step), ordering):
                continue
            if (s,step) not in ordering:
                ordering.append((s,step))
        steps = plan.steps+[s]
    
        agenda = plan.agenda + [(g,s) for g in imp.preconditions]
    
        # It's essential to calculate threats before adding the new link
        # (or plan_threats won't work). But the new order constraints are required
        # (e.g. so it won't detect a conflict with 'start'!)
        new_plan = plan._replace(ordering=ordering)
        threats = plan_threats(new_plan, s, new_causal_link)
        print '   '*depth+str(threats)
    
        new_plan = plan._replace(agenda=agenda,causal=causal,ordering=ordering,steps=steps)
        print '   '*depth+"add step", s, ":", new_plan.steps
        
        plans += resolve_threats(new_plan, threats, depth+1)

    return plans

def resolve_threats(plan, threats, depth):
    #print 'resolve_threats'
    assert isinstance(plan, Plan)
    #pp_plan(plan)
    #print 'R',

    valid_orderings = resolve_threats_(plan, plan.ordering, threats, depth)

    plans = [plan._replace(ordering=o) for o in valid_orderings]
    
    final_plans = []
    for new_plan in plans:
        final_plans += dfs_solve_precondition(new_plan, depth + 1)
    return final_plans

def resolve_threats_(plan, ordering, threats, depth):
    assert isinstance(plan, Plan)
    assert isinstance(plan.ordering, list)
    assert isinstance(ordering,list)
    assert isinstance(threats, list)
    if len(threats) == 0:
        return [ordering]
    
    (threat, causal_link) = threats[0]
    (stepA, goal, stepB) = causal_link

    #import pdb; pdb.set_trace()
    print '   '*depth+'resolving threat', threat, (stepA, goal, stepB)
    
    # Demotion
    demotion = (threat, stepA)
    # Promotion
    promotion = (stepB, threat)

    #import pdb; pdb.set_trace()

    resolved_orderings = []
    for new_order in [demotion, promotion]:
        # Now, resolve threats in the new version of the plan.
        # Reject it if there is a cycle in the ordering
        if is_cyclic(new_order, ordering):
            continue
        new_ordering = ordering + [new_order]
        
        resolved_orderings += resolve_threats_(plan, new_ordering, threats[1:], depth)
    
    return resolved_orderings

#def find_threats(plan):
#    threats = []
#    ordering = plan.ordering
#
#    for causal_link in plan.causal:
#        for possible_threat in plan.steps:
#            if possible_threat == causal_link[2]:
#                continue
#            if threatens_link(possible_threat, causal_link, plan.ordering):
#                threats.append((possible_threat, causal_link))
#
#    return threats

def plan_threats(plan, new_step, new_link):
    #import pdb; pdb.set_trace()
    threats = []
    # The new step may threaten an existing causal link.
    for existing_causal_link in plan.causal:
        if threatens_link(new_step, existing_causal_link, plan.ordering):
            threats.append((new_step, existing_causal_link))
    # Any existing step may threaten the new causal link.
    for existing_step in plan.steps:
        if threatens_link(existing_step, new_link, plan.ordering):
            threats.append((existing_step,new_link))
    return threats

def threatens_link(step, causal_link, ordering):
    (stepA, goal, stepB) = causal_link
    # If we've already decided step should be before stepA or after stepB,
    # then they don't matter.
    if (step_before(step, stepA, ordering)
        or step_before(stepB, step, ordering) or step in [stepA, stepB]):
        return False
    
    action = get_action_from_step(step)
    imp = action2imp[action]            
    results = imp.postconditions
    if any(conflict(goal,r) for r in results):
        return True

    return False

def backward_state_plan(initial_state, goal_state):
    add_implication(Implication('start',frozenset([]),initial_state))
    
    steps = []
    agenda = set(goal_state)
    return backward_state_plan_loop(agenda, 0)

def backward_state_plan_loop(agenda, depth):
    # Base case: return a plan with no steps
    if len(agenda) == 0:
        return []
    
    if depth >= 7:
        return None
    
    # Add a new step
    for imp in implications:
        # Find which goals in the agenda, this action would achieve (if any)
        goals_achieved = set(imp.postconditions) & agenda
        if len(goals_achieved) == 0:
            continue
        
        action = imp.action
        
        # Obviously we don't want to use the goal as an action, but
        # using the start state is OK
        if action == 'end':
            continue
        
        # Don't allow an action that will break any goals on the agenda.
        if any(conflict(post,goal) for post in imp.postconditions for goal in agenda):
            continue

        # Remove all of the goals this action will achieve, and add all of the goals it needs.
        new_agenda = agenda - set(imp.postconditions)
        new_agenda = new_agenda | set(imp.preconditions)
        
        #print '   '*depth+action,'achieves',goals_achieved,'requires',new_agenda
        print '   '*depth+action
        # Now recurse to find all of the actions to perform before this one.
        # Tacky special case
        if action == 'start':
            if len(new_agenda) == 0:
                earlier_actions = []
            else:
                continue
        else:
            earlier_actions = backward_state_plan_loop(new_agenda, depth+1)
        # If there are no earlier actions that will work, then continue the loop and try a different action.
        if earlier_actions == None:
            continue
        else:
            return earlier_actions + [action]
        
    return None

def forward_state_plan(initial_state, goal_state):
    steps = []
    state = set(initial_state)
    return forward_state_plan_loop(state, goal_state, 0)

def forward_state_plan_loop(state, goal_state, depth):
    # If the goal state is a subset of the current state (i.e. all goals are achieved)
    if goal_state <= state:
        return []
    
    if depth >= 7:
        return None
    
    # Add a new step
    for imp in implications:
        action = imp.action
        
        # Only allow an action if all of its preconditions are true in the current state.
        if set(imp.preconditions) <= state:
            # Add all the atoms that are true. Remove all the atoms that are false.
            new_state = set(state)
            for atom in imp.postconditions:
                if atom.startswith('~'):
                    new_state.remove(atom[1:])
                else:
                    new_state.add(atom)
            
            #print '   '*depth+action,'achieves',goals_achieved,'requires',new_agenda
            print '   '*depth+action
            later_actions = forward_state_plan_loop(new_state, goal_state, depth+1)
            # If there are no earlier actions that will work, then continue the loop and try a different action.
            if later_actions == None:
                continue
            else:
                return [action] + later_actions
        
    return None

def sussman_test():
    initial_state = frozenset(["on(C,A)","handempty","ontable(A)","ontable(B)","clear(B)","clear(C)"])
    #goal_state = frozenset(["on(A,B)","on(B,C)"])
    goal_state = frozenset(["on(A,B)","on(B,C)"])

#    plans = pop(initial_state, goal_state)
#    for plan in plans:
#        print plan.agenda
#        print choose_total_order(plan)
#        #import pdb; pdb.set_trace()
#
#    print 'plans tried per layer:', layer_sizes
#    print 'complete plans per layer:', complete_plans_in_layer

#    steps = backward_state_plan(initial_state, goal_state)
    steps = forward_state_plan(initial_state, goal_state)
    print steps

if __name__ == '__main__':
    sussman_test()

#order = [('start_0', 'end_1'), ('start_0', 'stack(A,B)_2'), ('stack(A,B)_2', 'end_1'), ('start_0', 'stack(B,C)_3'), ('stack(B,C)_3', 'end_1'), ('start_0', 'pickup(A)_4'), ('pickup(A)_4', 'end_1'), ('start_0', 'stack(A,B)_2'), ('start_0', 'pickup(B)_5'), ('pickup(B)_5', 'end_1'), ('stack(A,B)_2', 'pickup(B)_5'), ('start_0', 'stack(B,C)_3'), ('start_0', 'pickup(A)_4'), ('start_0', 'pickup(A)_4'), ('pickup(A)_4', 'pickup(B)_5'), ('stack(A,B)_2', 'pickup(A)_4'), ('start_0', 'pickup(B)_5'), ('stack(B,C)_3', 'pickup(B)_5'), ('stack(A,B)_2', 'stack(B,C)_3'), ('stack(B,C)_3', 'pickup(B)_5'), ('pickup(A)_4', 'stack(B,C)_3')]

#print choose_total_order(order)