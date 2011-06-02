;
; Action planner
;
; @author Zhenhua Cai <czhedu@gmail.com>
; @date   2011-05-31
;

; TODO: create and AndSeq link, inherits from ordered link for Actions

; Return a random member of the given list, 
; return an empty list, if the given list is empty. 
(define (random_select selection_list)
    (if (null? selection_list) 
        (list) 

        (list-ref selection_list
            (random (length selection_list) ) 
        )    
    ) 
)

; Return precondition/ goal given a psi rule
(define (get_psi_precondition rule)
    (list-ref (cog-outgoing-set rule) 0)
)

(define (get_psi_goal rule)
    (list-ref (cog-outgoing-set rule) 1) 
)

; Return #t if given atom is a Link    
(define (cog-link? atom)
    (cog-subtype? 'Link (cog-type atom) ) 
)    

; Return #t if given atom is an Node
(define (cog-node? atom)
    (cog-subtype? 'Node (cog-type atom) ) 
)

; Return a BindLink used by pattern matcher to search psi rules given goal
(define (find_psi_rule goal)
    (BindLink
        (VariableNode "$var_any")

        (ImplicationLink
            ; Pattern to be searched
            (ImplicationLink
                (VariableNode "$var_any")
                goal
            )

            ; Return value
            (ImplicationLink
                (VariableNode "$var_any")
                goal
            ) 
        )

    ); BindLink 
)

; Return a BindLink used by pattern matcher to search the action with suitable 
; groundings of all the variables that make the context True
;
; @param variables Usually it is the first outgoing of ForAllLink. 
;                  It may be a single VariableNode, a single TypedVariableLink
;                  or a ListLink of VariableNodes or TypedVariableLink. 
;  
; @note There's a chance that the action is empty. Especially in rules about 
;       relations, such as SomeOneKickYou => YouGetAngry  
;
(define (find_psi_action variables context action)
    (BindLink
        variables

        (ImplicationLink
            context

            (ListLink
                context
                action
            )
        )
    )
)

; Split an atom into context and action,
; the return value is 
;
; (list context_atom action_atom)
;
(define (split_context_action atom)
    (if (cog-atom? atom)
        ; If the input is an atom   
        (begin
            (let ( (atom_type (cog-type atom) )
                 )

;                 (display "Get an Atom ") (newline) (display atom) (newline)

                 (cond 
                     ; If the input is an ExecutionLink, return the input as action
                     ( (equal? atom_type 'ExecutionLink)
;                       (display "Found an Action") (newline) (display (list (list) (list atom) )) (newline)
                       (list (list) atom)
                     )

                     ; If the input is a Node but NOT a VariableNode, return the input as context 
                     ( (cog-subtype? 'Node atom_type)
;                       (display "Found a Node ") (newline) (display (list (list atom) (list) )) (newline)
                       (list atom (list) )
                     )

                     ; If the input is a Link, process its outgoings recursively
                     (else

                       (let  ( (atom_outgoings (cog-outgoing-set atom) )
                               (context_outgoings (list) )
                               (action_outgoings (list) )
                             ) 
                        
;                             (display "Get a Link ") (newline) (display atom) (newline)

                             ; Split the outgoings into context and action
                             (map-in-order
                                 (lambda (outgoing)                                    
                                     (let* ( (split_result (split_context_action outgoing) )
                                             (context (list-ref split_result 0) )
                                             (action (list-ref split_result 1) )
                                           )
                                          
                                           (if (not (null? context) )
                                               (set! context_outgoings
                                                     (append context_outgoings (list context) )
                                               )
                                           )

                                           (if (not (null? action) )
                                               (set! action_outgoings
                                                     (append action_outgoings (list action) )
                                               )
                                           )
                                     ) 
                                 ); lambda
  
                                 atom_outgoings
  
                             ); map-in-order
 
;                             (display "Get outgoings ") (newline) (display atom_outgoings) (newline)
;                             (display "Get context outgoings ") (newline) (display context_outgoings) (newline)
;                             (display "Get action outgoings ") (newline) (display action_outgoings) (newline)

                             ; Create and return the context and action
                             (list

                                 ; Build context atom
                                 (cond
                                     ( (null? context_outgoings)
                                       (list) 
                                     )

                                     ( (and (or (equal? atom_type 'AndLink)
                                                (equal? atom_type 'OrLink)
                                            )
                                            (equal? (length context_outgoings) 1)
                                       )

                                       (car context_outgoings)
                                     )

                                     (else
                                         (apply cog-new-link
                                             (append (list atom_type)
                                                      context_outgoings  
                                                     (list (cog-tv atom) (cog-av atom) )
                                             ) 
                                         )
                                     )
                                 ); cond
  
                                 ; Build action atom
                                 (cond
                                     ( (null? action_outgoings)
                                       (list) 
                                     )

                                     ( (and (or (equal? atom_type 'AndLink)
                                                (equal? atom_type 'OrLink)
                                            )
                                            (equal? (length action_outgoings) 1)
                                       )

                                       (car action_outgoings)
                                     )

                                     (else
                                         (apply cog-new-link
                                             (append (list atom_type)
                                                      action_outgoings  
                                                     (list (cog-tv atom) (cog-av atom) )
                                             ) 
                                         )
                                     )
                                 ); cond

                             ); list
      
                       ); let

                     ); else

                 ); cond

            ); let
        ); begin

        ; If the input is NOT an atom, return empty list
        (list (list) (list) )

    ); if
); define    

; Calculate the truth value of the given atom
; It simply use the truth value the atom and its outgoings already holds.
; It will NOT try to ground the variables. So if there are variables in the atom 
; or its outgoings recursively, using cog-bind-crisp instead to ground the variables
(define (cal_truth_value atom)
    (let* ( (atom_type (cog-type atom) )
            (atom_outgoings (cog-outgoing-set atom) )
          )
 
          (cond
              ; If the outgoings is empty, return the truth value 
              ; of the current atom
              ( (null? atom_outgoings)
                (get_truth_value_mean (cog-tv atom) )
              ) 
 
              ; If it is an AndLink, the truth value is the 
              ; average of all the truth values of its outgoings
              ( (equal? atom_type 'AndLink)
 
                (let ( (outgoing_truth_values 
                           (map cal_truth_value atom_outgoings) 
                       )
                     )
                     
                     (/ (apply + outgoing_truth_values)
                        (length outgoing_truth_values) 
                     )   
                )
              )
 
              ; If it is an OrLink, the truth value is the maximum 
              ; value of all the truth values of its outgoings
              ( (equal? atom_type 'OrLink)
 
                (max
                    (map cal_truth_value atom_outgoings) 
                )
              )
 
              ; If it is an NotLink, the truth value is 1 minus 
              ; the maximum value of all the truth values of its 
              ; outgoings
              ( (equal? atom_type 'NotLink)
 
                (- 1
                   (max
                       (map cal_truth_value atom_outgoings) 
                   )  
                )
              )
 
              ; If it is of other types, simply return the truth value
              (else
                (get_truth_value_mean (cog-tv atom) )
              ) 
 
          ); cond
 
    );let* 

); define

; Check if the context of the given psi rule is true. 
;
; @param precondition Usually the first outgoing of an psi rule, i.e. ImplicationLink
;
; @return A list as follows, 
;
;         (list
;             (list true_context_1 action_1)
;             (list true_context_2 action_2)
;             ...
;         )    
;
; @note If the context is true, then all the variables in returned context and
;       action would be properly grounded. 
;
;       If the context is empty, which means the agent can take the action at 
;       any situations, then this function will return an empty context and 
;       the action, if there exists
;
;       If the action is empty, which means the context will result some outcome, 
;       without the agent taking any action, then this function will return the
;       context and an empty action. 
;
(define (is_psi_context_true rule)
    (let* ( (precondition 
                (list-ref (cog-outgoing-set rule) 0)
            )
            (context_atom_list (split_context_action precondition) )
            (context (list-ref context_atom_list 0) )
            (action (list-ref context_atom_list 1) )
            (variables (list) )
          ) 

          (if (null? context)
              ; If the context is empty, which implies the agent can do the 
              ; action in any context, then return the action if there exists any
              (if (null? action) 
                  (list)
                  (list 
                      (list context action)
                  )    
              )

              ; If the context is NOT empty, then we should return the action 
              ; only when the context is true
              (begin
                  ; Get the variables of the rule,
                  ; the first step is to get a ForAllLink, ExistsLink or AverageLink 
                  ; contaning the rule, and then return the first outgoing of the link 
                  (let search_variables ( (incomings (cog-incoming-set rule) )
                                        ) 
                       (if (not (null? incomings) )
                           (let* ( (first_incoming (car incomings) )
                                   (first_incoming_type (cog-type first_incoming) )
                                 )
                                
                                 (if (or (equal? first_incoming_type 'ForAllLink)
                                         (equal? first_incoming_type 'ExistsLink)
                                         (equal? first_incoming_type 'AverageLink)
                                     )

                                     (set! variables 
                                           (list-ref (cog-outgoing-set first_incoming) 0)
                                     )

                                     (search_variables (cdr incomings) )
                                 ); if

                           ); let* 
                       ); let
                  );let 

;                  (display "Get variables") (newline) (display variables) (newline)

                  (if (null? variables)

                      ; If there's no variable in the psi rule, calculate the truth value
                      ; of context directly
                      (if (> (cal_truth_value context) 
                              0.5
                          )
                          (list (list context action) )
                          (list)
                      )

                      ; If there are variables in the psi rule, use pattern matcher to
                      ; find suitable groundings that makes the context True
                      (map unpack_query_result 
                          (query_atom_space_crisp (find_psi_action variables context action) )
                      )   
                      
                  ); if
                  ); begin
          ); if (null? context)

    ); let*
); define

; Make a plan for a given goal
;
; @return a list of (context action) and corresponding rules that would lead to
;         the given goal, like
;
;         (list 
;             (list
;                 solved_rule_1
;                 solved_rule_2
;                 ...
;             )
;
;             (list
;                 (list context_1 action_1)
;                 (list context_2 action_2)
;                 ...
;             )   
;         )
;
; @note This function will only return a list of actions that can be performed
;       immediately in current cognitive cycle. That means it will not preserve 
;       a complete chain of actions that would lead to the goal, if some 
;       preconditions of rules are not satisfied right now. It that situation, 
;       this function returns a bunch of actions that would be helpful for the
;       goal. 
;
;       For example there are two rules as below:
;
;       RandomSearchAction => GetFoodGoal
;       GetFoodGoal AND EatFoodAction => EnergyDemandGoal
;   
;       If there's food nearby now, this function will only return the action 
;       EatFoodAction. 
;
;       If there's no food nearby now, this function will only return the action 
;       RandomSearchAction.Since the world is always changing and there's chance
;       that the agent failes to get the food after it randomly search for a
;       while, it will not return EatFoodAction. 
;
; @todo Currently you should use the same variables in different rules if they 
;       have the same meaning. For instance, in the previous example, you should 
;       stick to (VariableNode "$var_food") in both two rules. We will remove this 
;       limitation later. Probably just add a new function like get_psi_rules.   
;
(define (make_psi_plan goal)
    (let ( (planned_goal_list (list) )
           (reachable_rule_list (list) )
           (unreachable_rule_list (list) )
           (context_action_list (list) )
           (solved_rule_list (list) )
         )

         (let do_plan ( (current_goal goal)
                      )

             ; If current goal is not empty and it has not been planned yet,
             ; make a plan for it
             (if (and (not (null? current_goal) )
                      (equal? (member current_goal planned_goal_list) #f)
                 )    

                ; Get all the rules related to current goal 
                (let ( (available_rule_list
                           (query_atom_space (find_psi_rule current_goal) ) 
                       )
                     ) 

;                    (display "do_plan for ") (newline) (display current_goal) (newline)

                    ; Append current goal to end of the planned goal list,
                    ; then it will not be planned twice. 
                    (set! planned_goal_list 
                          (append planned_goal_list (list current_goal) )
                    )

                    (if (null? available_rule_list)
                        ; If there's no rule directly attach to current goal, 
                        ; we should dive into its outgoings             
                        (let ( (atom_type (cog-type current_goal) )
                               (atom_outgoings (cog-outgoing-set current_goal) )
                             )

;                              (display "No direct rules for ") (newline) (display current_goal) (newline)
                            
                             (if (not (null? atom_outgoings) )
                                 (cond 
                                     ( (equal? atom_type 'AndLink)
                                       (map do_plan atom_outgoings)
                                     ) 
  
                                     ( (equal? atom_type 'OrLink)
                                       (do_plan
                                           (random_select atom_outgoings) 
                                       )
                                     )    
                                 ); cond
                             ); if
                        ); let
                     
                        ; If there are rules attaching to the current goal
                        (let ( (selected_rule (list) )
                               (search_result (list) )
                             )

;                              (display "Available rules ") (newline) (display available_rule_list) (newline)
                             
                             ; Search a rule with True context 
                             (let search_rule_with_true_context ( (rule_list available_rule_list)
                                                                )

                                  (if ( not (null? rule_list) )
                                      (let ( (current_rule (car rule_list) )
                                           )

                                           (if (equal? (member current_rule unreachable_rule_list)
                                                       #f
                                               )
                                              
                                               (begin 
                                                   ; Search
                                                   (set! search_result 
                                                         (is_psi_context_true current_rule)
                                                   ) 
                           
                                                   (if (null? search_result)
                                                       ; Not found
                                                       (begin
                                                           (set! unreachable_rule_list 
                                                                 (append unreachable_rule_list (list current_rule) ) 
                                                           )
;                                                            (display "search_result ") (newline) (display search_result) (newline)
;                                                            (display "False context rule ") (newline) (display current_rule) (newline)
                                                           (search_rule_with_true_context (cdr rule_list) ) 
                                                       )   

                                                       ; Found
                                                       (begin
                                                           (set! reachable_rule_list
                                                                 (append reachable_rule_list (list current_rule) )
                                                           )
;                                                            (display "search_result ") (newline) (display search_result) (newline)
;                                                            (display "True context rule ") (newline) (display current_rule) (newline)
                                                           (set! selected_rule current_rule)
                                                       )   
                                                   ); if 
                                               ); begin

                                           ); if 

                                      ); let
                                  ); if
                             ); let
  
                             (if (null? selected_rule)
                                 ; If there's no rule with True context, randomly select a rule
                                 ;
                                 ; TODO: we should take advantage of weights (truth value) of
                                 ;       rules for selection using roulette wheel selection. 
                                 (begin
;                                      (display "Failed to find any rule with true context ") (newline)

                                     (set! selected_rule 
                                           (random_select available_rule_list)
                                     )
 
                                     (let* ( (precondition (get_psi_precondition selected_rule) )
                                             (split_result (split_context_action precondition) )
                                             (context (list-ref split_result 0) )
                                           ) 
                                           (do_plan context)
                                     )
                                 )
   
                                 ; If found a rule with True context
                                 (let ( (selected_context_action (random_select search_result) )
                                      ) 

;                                       (display "Found a rule with true context ") (display search_result) (newline)

                                      (if (equal? (member selected_context_action context_action_list)
                                                  #f
                                          )

                                          (begin
                                              (set! context_action_list 
                                                    (append context_action_list
                                                            (list selected_context_action)
                                                    )
                                              )

                                              (set! solved_rule_list
                                                    (append solved_rule_list
                                                            (list selected_rule)
                                                    ) 
                                              )
                                          ); begin
                                      )
                                 ); let
   
                             ); if (null? selected_rule)
                        ); let
                    ); if (null? available_rule_list)

                ); let
             ); if

         ); let do_plan

         ; Return the result of planning
         (if (and (null? solved_rule_list)
                  (null? context_action_list)
             )
             
             (list)

             (list solved_rule_list context_action_list)
         )

    ); let    
); define

; BindLink used by pattern matcher to get the ReferenceLink given the first
; outgoing
(define (find_reference_link first_outgoing)
    (BindLink
        ; Variables
        (VariableNode "$var_any")

        (ImplicationLink
            ; Pattern to be searched
            (ReferenceLink
                first_outgoing
                (VariableNode "$var_any")
            ) 

            ; Return result
            (ReferenceLink
                first_outgoing
                (VariableNode "$var_any")
            ) 
        ); ImplicationLink

    ); BindLink 
)

; Return the ReferenceLink given the first outgoing
(define (get_reference_link first_outgoing)
    (let ( (reference_link_list 
                (query_atom_space (find_reference_link first_outgoing) ) 
           )
         )

         (if (null? reference_link_list) 
             (list)
             (car reference_link_list)
         )
    ) 
)

; Return the reference of the given atom
(define (get_reference first_outgoing)
    (let ( (reference_link (get_reference_link first_outgoing) )
         )
       
         (if (null? reference_link)
             (list) 
             (list-ref (cog-outgoing-set reference_link) 1)
         )   
    ) 
)

; Remove the old ReferenceLink (if any) holding the first_outgoing and 
; create a new ReferenceLink containing the given outgoings, 
; return the newly created ReferenceLink
(define (update_reference_link first_outgoing second_outgoing)
    (let ( (old_reference_link (get_reference_link first_outgoing) )
         )

         (if (not (null? old_reference_link) )
             (cog-delete old_reference_link) 
         )

         (ReferenceLink
             first_outgoing
             second_outgoing
         )
    )
)

; Return a list containing all the demand goals (EvaluationLink) 
; The list of demand goals is initialized by PsiActionSelectionAgent::initDemandGoalList
(define (get_demand_goal_list)
    (let ( (demand_goal_list_list_link
               (get_reference (ConceptNode "plan_demand_goal_list") ) 
           ) 
         )

         (cog-outgoing-set demand_goal_list_list_link)
    )
)

; Return a randomly selected demand goal (EvaluationLink)
(define (get_random_demand_goal) 
    (random_select (get_demand_goal_list) )
)

; Reuturn the demand goal (EvaluationLink) with lowest truth value
(define (get_most_critical_demand_goal)
    (let ( (demand_goal_list (get_demand_goal_list) )
           (most_critical_demand_goal (list) )
           (most_critical_truth_value 1)
         )
        
         (map 
             (lambda (demand_goal)
                 (let ( (demand_goal_truth_value
                             (get_truth_value_mean (cog-tv demand_goal) )
                        )
                      )

                      (if (< demand_goal_truth_value most_critical_truth_value)
                          (begin
                              (set! most_critical_truth_value demand_goal_truth_value) 
                              (set! most_critical_demand_goal demand_goal)
                          )    
                      )
                 ) 
             )

             demand_goal_list
         )
       
         ; Return the demand goal with lowest truth value
         most_critical_demand_goal
    )
)

; Do the planning, this function is called by PsiActionSelectionAgent::doPlanning
; TODO: take advantage of modulators later
(define (do_planning)
    (let ( (selected_demand_goal (list) )
           (plan_result (list) )
           (rule_list (list) )
           (context_list (list) )
           (action_list (list) )
         )

         ; Select the most critical demand goal
         ; TODO: if selected_demand_goal is null?
         (set! selected_demand_goal (get_most_critical_demand_goal) )

         ; Planning
         ; TODO: if the planning result is null?
         (set! plan_result (make_psi_plan selected_demand_goal) )

         ; Get rule list
         (set! rule_list (car plan_result) )

         ; Get context and action list
         ; replace empty (list) with NULL_CONTEXT or NULL_ACTION
         (map-in-order
             (lambda (context_action)
                 (let ( (context (list-ref context_action 0) )
                        (action (list-ref context_action 1) )
                      )

                      (if (null? context)
                          (set! context NULL_CONTEXT) 
                      )

                      (if (null? action)
                          (set! action NULL_ACTION) 
                      )

                      (set! context_list
                            (append context_list (list context) )
                      )

                      (set! action_list
                            (append action_list (list action) )
                      )
                 ); let 
             ); lambda 

             (list-ref plan_result 1)
         )

         ; Update the planning result stored in AtomSpace
         ; PsiActionSelectionAgent will actually execute the plan
         (update_reference_link 
             (ConceptNode "plan_rule_list") 
             (apply ListLink rule_list) 
         )

         (update_reference_link
             (ConceptNode "plan_context_list") 
             (apply ListLink context_list)
         )

         (update_reference_link
             (ConceptNode "plan_action_list") 
             (apply ListLink action_list)
         )

         (update_reference_link
             (ConceptNode "plan_selected_demand_goal") 
             selected_demand_goal
         )

    ); let        
)
