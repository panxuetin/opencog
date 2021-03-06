/*
 * opencog/embodiment/Control/Procedure/RunningComboProcedure.h
 *
 * Copyright (C) 2002-2009 Novamente LLC
 * All Rights Reserved
 * Author(s): Nil Geisweiller, Moshe Looks
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _RUNNING_PROCEDURE_H
#define _RUNNING_PROCEDURE_H

/**
 * encapsulates a procedure currently running inside some ComboInterpreter
 *  (that returns an action result - non-action procs should be run via the
 *  combo::eval or combo::eval_throws functions)
 *
 * do YOU have any procedures running? then you'd better go to catch them!
 *
 * for general design documentation, see the section "Interpreter Design" in
 *  the document Embodiment Combo Dialect:
 *  http://wiki.opencog.org/w/EmbodimentCombo_%28Embodiment%29#Interpreter_Design
 */

#include <opencog/embodiment/Control/PerceptionActionInterface/PAI.h>

#include <opencog/comboreduct/combo/vertex.h>
#include <opencog/comboreduct/combo/type_tree.h>
#include <opencog/comboreduct/combo/eval.h>

#include <stack>
#include <exception>
#include <boost/logic/tribool.hpp>

#include <opencog/util/exceptions.h>
#include <opencog/embodiment/WorldWrapper/WorldWrapper.h>

namespace opencog { namespace Procedure {

using namespace pai;
using namespace world;

struct RunningComboProcedure : public combo::Evaluator {

    //the exception that
    //TODO
    struct ActionPlanSendingFailure {
        ActionPlanSendingFailure(ActionPlanID _id) : id(_id) { }
        ActionPlanID id;
    };

    //these implement the callback interface combo::Evaluator used by the
    //regular combo interpreter when called by us
    combo::vertex eval_action(combo::combo_tree::iterator,
                              combo::variable_unifier&);
    combo::vertex eval_procedure(combo::combo_tree::iterator,
                                 combo::variable_unifier&);
    combo::vertex eval_percept(combo::combo_tree::iterator,
                               combo::variable_unifier&);
    combo::vertex eval_indefinite_object(combo::indefinite_object,
                                         combo::variable_unifier&);

    //construct an rp from a worldwrapper and a tree
    RunningComboProcedure(WorldWrapperBase& ww, const combo::combo_tree& tr,
                          const std::vector<combo::vertex>& arguments,
                          bool doesSendDefinitePlan = true,
                          combo::variable_unifier& vu = combo::variable_unifier::DEFAULT_VU());

    //copy ctor - fatal runtime error if the rhs has already begun running
    RunningComboProcedure(const RunningComboProcedure&);

    //each call to cycle executes a single action plan (if ready)
    //throws if execution of the action plan (PAI::sendActionPlan(ActionPlanID)) fails
    //TODO : above is not true anymore, find the equivalent
    void cycle() throw(ActionPlanSendingFailure, AssertionException, std::bad_exception);

    //terminate - prevent future plans from being evaluated,
    //sets result to null_vertex - note that this is not the same as failure
    void stop() {
        _tr = combo::combo_tree(combo::id::null_vertex);
        _it = _tr.end();
    }

    //is the rp ready to run an action plan?
    bool isReady() const {
        return (_tr.is_valid(_it) && (!_hasBegun || _ww.isPlanFinished()));
    }
    //is the rp done running?
    bool isFinished() {
        if (!finished) {
            finished = (!_tr.is_valid(_it) && (!_hasBegun || (!_planSent || _ww.isPlanFinished())));
        }
        return finished;

        /*      stringstream ss;
              ss << _tr;
              logger().debug(
              "RunningComboProcedure - '%s' is_valid '%s', has begun '%s', plan sent '%s', plan finished '%s'",
              ss.str().c_str(), _tr.is_valid(_it)?"true":"false", _hasBegun?"true":"false", _planSent?"true":"false", _ww.isPlanFinished()?"true":"false");

              return (!_tr.is_valid(_it) &&
               (!_hasBegun || (!_planSent || _ww.isPlanFinished())));
          */
    }
    //did the last action plan executed by the procedure fail?
    bool isFailed() const {
        return
            (_failed ? true :
             (!_failed ? false :
              (_hasBegun && _ww.isPlanFailed())));
    }

    // Get the result of the procedure
    // Can be called only if the following conditions are true:
    // - procedure execution is finished (checked by isFinished() method)
    // returns null_procedure if execution was stopped in the middle
    combo::vertex getResult() {
        OC_ASSERT(isFinished(), "RunningComboProcedure - Procedure isn't finished.");
        if (_hasBegun)
            return isFailed() ? combo::id::action_failure : combo::id::action_success;
        if (_tr.size() == 1)
            return *_tr.begin();
        return combo::id::action_success;
    }

    combo::variable_unifier& getUnifierResult() {
        OC_ASSERT(isFinished(), "RunningComboProcedure - Procedure isn't finished.");
        return _vu;
    }

protected:
    typedef combo::combo_tree::sibling_iterator sib_it;

    WorldWrapperBase& _ww;

    combo::combo_tree _tr;
    
    /** _it is kept pointing at the tree node to be executed next; an
     * invalid iterator indicates nothing more to execute. for an
     * action_action_if conditional, pointing at the root of the
     * first child indicates that the condition branch has been
     * fully evaluated but not acted on yet (whereas pointing at the
     * conditional node itself indicates that the condition branch
     * has not yet been evaluated.
     */
    sib_it _it;

    bool _hasBegun; //have we started an plan yet?
    bool _planSent;
    boost::tribool _failed; //set to true if failed, false if not failed,
    //unknown to decide by query
    //a stack is needed to handle nested loops correctly
    std::stack<std::pair<combo::combo_tree::sibling_iterator, combo::vertex> > _stack;
    bool _inCompound; //used for handling builtin compound functions (e.g. follow)

    //used for sending action plans
    bool exec(sib_it x, combo::variable_unifier& vu) {
        return execSeq(x, ++sib_it(x), vu);
    }
    bool execSeq(sib_it, sib_it, combo::variable_unifier& vu);

    /// @return true iff an action plan gets executed
    bool beginCompound();
    /// @return true iff an action plan gets executed
    bool continueCompound();

    /// update _it to go to the next point of execution
    void moveOn();

    /// for evaluating procedures inplace
    void expand_procedure_call(combo::combo_tree::iterator) throw
        (ComboException, AssertionException, std::bad_exception);
    void expand_and_evaluate_subtree(combo::combo_tree::iterator it, combo::variable_unifier&);

private:
    /// initialization - only called from ctors
    void init(const std::vector<combo::vertex>&);

    /**
     * true if the combo interpreter
     * evaluates the indefinite aguments
     * of a plan during interpretation
     * (that is the plan once sent contains only
     * definite objects)
     * false if the combo interpreter
     * sends directly unevaluated indefinite objects
     * in the plan
     * this is escentially used by NoSpaceLife
     * in order to deal with the random operators
     * optimization, to avoid Monte Carlos
     * simulations
     */
    bool _doesSendDefinitePlan;

    // unifier that holds options to be tested when a wild card character "_*_" is
    // used in a combo script
    combo::variable_unifier _vu;

    bool finished;
};

}} // ~namespace opencog::Procedure

#endif
