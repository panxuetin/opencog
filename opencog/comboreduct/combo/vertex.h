/*
 * opencog/comboreduct/combo/vertex.h
 *
 * Copyright (C) 2002-2008 Novamente LLC
 * All Rights Reserved
 *
 * Written by Nil Geisweiller
 *            Moshe Looks
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
#ifndef _COMBO_VERTEX_H
#define _COMBO_VERTEX_H

#include <boost/functional/hash.hpp>
#include <boost/variant.hpp>
// #include <boost/range.hpp>

#include <opencog/util/tree.h>
#include <opencog/util/numeric.h>
#include <opencog/util/exceptions.h>
#include <opencog/util/oc_assert.h>

#include "argument.h"
#include "action.h"
#include "builtin_action.h"
#include "action_symbol.h"
#include "perception.h"
#include "definite_object.h"
#include "indefinite_object.h"
#include "message.h"
#include "ann.h"
#include "common_def.h"

// uncomment that if you want to interpret log(x) as log(abs(x))
// #define ABS_LOG

namespace opencog { namespace combo {

class procedure_call_base;

typedef const procedure_call_base* procedure_call;

//this idiom allows builtin to live in namespace combo, but all
//ids to be hidden inside the namespace id
namespace id {

 // Nil: IMO we should replace these enum by class containing the enum
 // + methods with properties and strings rather than having that in
 // description. And of course do the same for builtin actions and
 // other enum. That way there would no possible ambiguity in
 // overloading methods which interpret enum as int by default which
 // leads to nightmarish errors. It might also that the new standard
 // C++0x fixes that that problem all by itself since in that standard
 // 2 different enum are considered as different type.
enum builtin {
    null_vertex = 0,
    logical_true, logical_false, // contants are put first to be in
                                 // sync with
                                 // lexicographic_subtree_order, it is
                                 // not mandatory but it should make
                                 // the reduct engine a bit faster
    logical_and, logical_or, logical_not, 
    contin_if,
    boolean_if,
    plus, times, div, exp,
    log, // if ABS_LOG is enabled then log(x) := log(abs(x))
    sin,
    greater_than_zero,
    impulse,
    rand,         //random contin_t in [0,1) FIXME TODO : update reduct rules
    builtin_count //to get the number of builtin
};

}


typedef id::builtin builtin;

//this idiom allows wild_card to live in namespace combo, but all
//ids to be hidden inside the namespace id
namespace id {
enum wild_card {
    asterisk = 0,
    wild_card_count //to get the number of wild_card
};
}
typedef id::wild_card wild_card;

typedef double contin_t;

// contants are put first to be in sync with
// lexicographic_subtree_order, it is not mandatory but it should make
// the reduct engine a bit faster
typedef boost::variant < contin_t,
                         builtin,
                         wild_card,
                         argument,
                         action,
                         builtin_action,
                         perception,
                         definite_object,
                         indefinite_object,
                         message,
                         procedure_call,
                         action_symbol,
                         ann_type > vertex;

typedef std::set<vertex> vertex_set;
typedef vertex_set::iterator vertex_set_it;
typedef vertex_set::const_iterator vertex_set_const_it;

typedef std::vector<vertex> vertex_seq;
typedef vertex_seq::iterator vertex_seq_it;
typedef vertex_seq::const_iterator vertex_seq_const_it;
        
typedef std::set<vertex> operator_set;
typedef operator_set::iterator operator_set_it;
typedef operator_set::iterator operator_set_const_it;

typedef std::vector<vertex> argument_list;
typedef argument_list::iterator argument_list_it;
typedef argument_list::const_iterator argument_list_const_it;
typedef std::vector<argument_list> argument_list_list;
typedef argument_list_list::iterator argument_list_list_it;
typedef argument_list_list::const_iterator argument_list_list_const_it;

// Disambiguate stream operator; use the one declared in util/tree.h
std::istream& operator>>(std::istream& in, combo::vertex& v);

// procedure_call == vertex
// bool operator==(const vertex& v, procedure_call h);

// inline bool operator==(procedure_call h, const vertex& v)
// {
//     return (v == h);
// }
// bool operator!=(const vertex& v, procedure_call h);
// bool operator!=(procedure_call h, const vertex& v);
        
// builtin == vertex
inline bool operator==(const vertex& v, builtin h)
{
    if (const builtin* vh = boost::get<builtin>(&v))
        return (*vh == h);
    return false;
}
inline bool operator==(builtin h, const vertex& v)
{
    return (v == h);
}
inline bool operator!=(const vertex& v, builtin h)
{
    return !(v == h);
}
inline bool operator!=(builtin h, const vertex& v)
{
    return !(v == h);
}

//wild_card == vertex
inline bool operator==(const vertex& v, wild_card w)
{
    if (const wild_card* vw = boost::get<wild_card>(&v))
        return (*vw == w);
    return false;
}
inline bool operator==(wild_card w, const vertex& v)
{
    return (w == v);
}
inline bool operator!=(const vertex& v, wild_card w)
{
    return !(v == w);
}
inline bool operator!=(wild_card w, const vertex& v)
{
    return !(v == w);
}

//action == vertex
inline bool operator==(const vertex& v, action a)
{
    if (const action* va = boost::get<action>(&v))
        return (*va == a);
    return false;
}
inline bool operator==(action a, const vertex& v)
{
    return (v == a);
}
inline bool operator!=(const vertex& v, action a)
{
    return !(v == a);
}
inline bool operator!=(action a, const vertex& v)
{
    return !(v == a);
}

//builtin_action == vertex
inline bool operator==(const vertex& v, builtin_action a)
{
    if (const builtin_action* va = boost::get<builtin_action>(&v))
        return (*va == a);
    return false;
}
inline bool operator==(builtin_action a, const vertex& v)
{
    return (v == a);
}
inline bool operator!=(const vertex& v, builtin_action a)
{
    return !(v == a);
}
inline bool operator!=(builtin_action a, const vertex& v)
{
    return !(v == a);
}

//perception == vertex
inline bool operator==(const vertex& v, perception p)
{
    if (const perception* vp = boost::get<perception>(&v))
        return (*vp == p);
    return false;
}
inline bool operator==(perception p, const vertex& v)
{
    return (v == p);
}
inline bool operator!=(const vertex& v, perception p)
{
    return !(v == p);
}
inline bool operator!=(perception p, const vertex& v)
{
    return !(v == p);
}

//definite_object == vertex
inline bool operator==(const vertex& v, definite_object d)
{
    if (const definite_object*
            vd = boost::get<definite_object>(&v))
        return (*vd == d);
    return false;
}
inline bool operator==(definite_object d, const vertex& v)
{
    return (v == d);
}
inline bool operator!=(const vertex& v, definite_object d)
{
    return !(v == d);
}
inline bool operator!=(definite_object d, const vertex& v)
{
    return !(v == d);
}

//indefinite_object == vertex
inline bool operator==(const vertex& v, indefinite_object i)
{
    if (const indefinite_object*
            vi = boost::get<indefinite_object>(&v))
        return (*vi == i);
    return false;
}
inline bool operator==(indefinite_object i, const vertex& v)
{
    return (v == i);
}
inline bool operator!=(const vertex& v, indefinite_object i)
{
    return !(v == i);
}
inline bool operator!=(indefinite_object i, const vertex& v)
{
    return !(v == i);
}

/*
//ann_ids == vertex
inline bool operator==(const vertex& v, ann_type a)
{
   if (const ann_ids* va = boost::get<ann_ids>(&v))
	return (*va == a);
   return false;
}
inline bool operator==(ann_ids a, const vertex& v)
{
   if (const ann_ids* va = boost::get<ann_ids>(&v))
	return (*va == a);
   return false;
}
inline bool operator!=(const vertex& v, ann_ids a)
{
    return !(v == a);
}
inline bool operator!=(ann_ids a,const vertex& v)
{ 
    return !(v == a);
}
*/

//message == vertex
inline bool operator==(const vertex& v, message m)
{
    if (const message* vm = boost::get<message>(&v))
        return (*vm == m);
    return false;
}
inline bool operator==(message m, const vertex& v)
{
    return (v == m);
}
inline bool operator!=(const vertex& v, message m)
{
    return !(v == m);
}
inline bool operator!=(message m, const vertex& v)
{
    return !(v == m);
}

//action_symbol == vertex
inline bool operator==(const vertex& v, action_symbol i)
{
    if (const action_symbol*
            vi = boost::get<action_symbol>(&v))
        return (*vi == i);
    return false;
}
inline bool operator==(action_symbol i, const vertex& v)
{
    return (v == i);
}
inline bool operator!=(const vertex& v, action_symbol i)
{
    return !(v == i);
}
inline bool operator!=(action_symbol i, const vertex& v)
{
    return !(v == i);
}

// don't know why this is needed *in namespace boost*, but it is, for
// e.g. calling a generic stl function that compares vertices for
// inequality
inline bool operator!=(const vertex& v1, const vertex& v2)
{
    return !(v1 == v2);
}

inline size_t hash_value(const message& m)
{
    /// WARNING: let the boost namespace as it permits to not generate
    /// infinit recursive calls of hash_value(const vertex& v)
    return boost::hash_value(m.getContent());
}

inline size_t hash_value(const vertex& v)
{
    using boost::hash_combine;

    static const size_t c1 = size_t(id::builtin_count);
    //it is likely that a combo will rarely have over 15 arguments
    static const size_t c2 = c1 + 15;
    static const size_t c3 = c2 + size_t(id::action_count);
    static const size_t c_last = c3;

    //there will be some overlap between contin_t, definite_object, message
    //and procedure_call but this overlap is unavoidable

    if (const builtin* h = boost::get<builtin>(&v))
        return size_t(*h);
    if (const wild_card* w = boost::get<wild_card>(&v))
        return size_t(*w);
    if (const argument* a = boost::get<argument>(&v))
        return size_t(a->idx * (a->is_negated() + 2)) + c1;
    if (const contin_t* c = boost::get<contin_t>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace as it permits to not generate
        /// infinit recursive calls of hash_value(const vertex& v)
        hash_combine(tmp, boost::hash_value(*c));
        return tmp;
    }
    if (const action* a = boost::get<action>(&v))
        return size_t(*a) + c2;
    if (const builtin_action* b = boost::get<builtin_action>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace (see above)
        hash_combine(tmp, boost::hash_value(*b));
        return tmp;
    }
    if (const perception* p = boost::get<perception>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace (see above)
        hash_combine(tmp, boost::hash_value(*p));
        return tmp;
    }
    if (const definite_object* d = boost::get<definite_object>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace (see above)
        hash_combine(tmp, boost::hash_value(*d));
        return tmp;
    }
    if (const indefinite_object* i = boost::get<indefinite_object>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace (see above)
        hash_combine(tmp, boost::hash_value(*i));
        return tmp;
    }
    if (const message* m = boost::get<message>(&v)) {
        size_t tmp = c_last;
        hash_combine(tmp, combo::hash_value(*m));
        return tmp;
    }
    if (const procedure_call* pc = boost::get<procedure_call>(&v)) {
        size_t tmp = c_last;
        std::cout << pc << std::endl;
        /// WARNING: let the boost namespace (see above)
        // TODO
        // hash_combine(tmp, boost::hash_value(*pc));
        return tmp;
    }
    if (const action_symbol* as = boost::get<action_symbol>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace (see above)
        hash_combine(tmp, boost::hash_value(*as));
        return tmp;
    }
    if (const ann_type* a = boost::get<ann_type>(&v)) {
        size_t tmp = c_last;
        /// WARNING: let the boost namespace (see above)
        hash_combine(tmp, boost::hash_value(a->idx));
        return tmp;
    }
    OC_ASSERT(false, "A case is missing");
    return 0;
}

typedef tree<vertex> combo_tree;
// ns stands for normal size
typedef std::set<combo_tree, size_tree_order<vertex> > combo_tree_ns_set;
typedef combo_tree_ns_set::iterator combo_tree_ns_set_it;
typedef combo_tree_ns_set::const_iterator combo_tree_ns_set_const_it;

// Disambiguate stream operator; use the one declared in util/tree.h
std::istream& operator>>(std::istream& in, combo::combo_tree& tr);

template<typename T>
inline bool is_associative(const T& v)
{
    return (v == id::logical_and || v == id::logical_or ||
            v == id::plus || v == id::times ||
            //actions
            v == id::sequential_and || v == id::sequential_or ||
            v == id::sequential_exec);
}
template<typename T>
inline bool is_commutative(const T& v)
{
    return (v == id::logical_and || v == id::logical_or ||
            v == id::plus || v == id::times
            || is_symmetric(v));
}

//properties of perceptions

template<typename T>
inline bool is_ultrametric(const T& v)
{
    if (is_perception(v))
        return get_perception(v)->is_ultrametric();
    else return false;
}

template<typename T>
inline bool is_transitive(const T& v)
{
    if (is_perception(v))
        return get_perception(v)->is_transitive();
    else return false;
}

template<typename T>
inline bool is_reflexive(const T& v)
{
    if (is_perception(v))
        return get_perception(v)->is_reflexive();
    else return false;
}

template<typename T>
inline bool is_irreflexive(const T& v)
{
    if (is_perception(v))
        return get_perception(v)->is_irreflexive();
    else return false;
}

template<typename T>
inline bool is_symmetric(const T& v)
{
    if (is_perception(v))
        return get_perception(v)->is_symmetric();
    else return false;
}

template<typename T>
inline bool is_identity_of_indiscernibles(const T& v)
{
    if (is_perception(v))
        return get_perception(v)->is_identity_of_indiscernibles();
    else return false;
}

inline bool is_procedure_call(const vertex& v)
{
    return (boost::get<procedure_call>(&v));
}

inline procedure_call get_procedure_call(const vertex& v)
{
    return (boost::get<procedure_call>(v));
}

inline bool is_action_symbol(const vertex& v)
{
    return (boost::get<action_symbol>(&v));
}

inline action_symbol get_action_symbol(const vertex& v)
{
    return (boost::get<action_symbol>(v));
}

inline bool is_indefinite_object(const vertex& v)
{
    return (boost::get<indefinite_object>(&v));
}

inline indefinite_object get_indefinite_object(const vertex& v)
{
    return (boost::get<indefinite_object>(v));
}

inline bool is_message(const vertex& v)
{
    return (boost::get<message>(&v));
}

inline message get_message(const vertex& v)
{
    return (boost::get<message>(v));
}

inline bool is_builtin(const vertex& v)
{
    return (boost::get<builtin>(&v));
}

inline builtin get_builtin(const vertex& v)
{
    return (boost::get<builtin>(v));
}

inline bool is_wild_card(const vertex& v)
{
    return (boost::get<wild_card>(&v));
}

inline wild_card get_wild_card(const vertex& v)
{
    return (boost::get<wild_card>(v));
}

inline bool is_contin(const vertex& v)
{
    return (boost::get<contin_t>(&v));
}
inline contin_t get_contin(const vertex& v)
{
    return (boost::get<contin_t>(v));
}

/**
 * return true if the vertex is an argument. Note, however, that
 * this does not take into account the type of the argument; thus,
 * any code that uses this blindly runs the risk of accepting arguments
 * of the wrong type (e.g. boolean instead of contin, oor v.v.).
 *
 * This should be fixed, but we don't have the infrastructure for this.
 */
inline bool is_argument(const vertex& v)
{
    return (boost::get<argument>(&v));
}
inline argument& get_argument(vertex& v)
{
    return boost::get<argument>(v);
}
inline bool is_ann_type(const vertex& v)
{
    return (boost::get<ann_type>(&v));
}
inline ann_type& get_ann_type(vertex& v)
{
    return (boost::get<ann_type>(v));
}
inline const argument& get_argument(const vertex& v)
{
    return boost::get<argument>(v);
}
inline bool is_negated(vertex& v)
{
    if (argument* a = boost::get<argument>(&v))
        return a->is_negated();
    return false;
}

inline bool is_action(const vertex& v)
{
    return (boost::get<action>(&v));
}

inline action get_action(const vertex& v)
{
    return (boost::get<action>(v));
}

inline bool is_builtin_action(const vertex& v)
{
    return (boost::get<builtin_action>(&v));
}
inline builtin_action get_builtin_action(const vertex& v)
{
    return (boost::get<builtin_action>(v));
}
inline bool is_action_result(const vertex& v)
{
    return (v == id::action_failure || v == id::action_success);
}

inline bool is_perception(const vertex& v)
{
    return (boost::get<perception>(&v));
}

inline perception get_perception(const vertex& v)
{
    return (boost::get<perception>(v));
}

inline bool is_definite_object(const vertex& v)
{
    return (boost::get<definite_object>(&v));
}

inline definite_object get_definite_object(const vertex& v)
{
    return (boost::get<definite_object>(v));
}

inline vertex bool_to_vertex(bool b)
{
    return (b ? id::logical_true : id::logical_false);
}
inline bool vertex_to_bool(const vertex& v)
{
    OC_ASSERT(v == id::logical_true || v == id::logical_false,
              "vertex should be 'id::logical_true' or 'id::logical_false'.");
    return (v == id::logical_true);
}

// Return logical_true if logical_false and vice versa
//
// Note: don't rename it negate as it enters in conflict with
// std::negate(string)
vertex negate_vertex(const vertex& v);

inline bool is_complement(const vertex& x, const vertex& y)
{
    if (const argument* ax = boost::get<argument>(&x)) {
        if (const argument* ay = boost::get<argument>(&y)) {
            return (ax->idx == -ay->idx);
        }
    }
    return false;
}

template<typename T>
inline bool is_boolean(const T& v)
{
    return (v == id::logical_true || v == id::logical_false);
}

template<typename T>
inline bool is_logical_operator(const T& v)
{
    return (v == id::logical_and || v == id::logical_or || v == id::logical_not);
}

template<typename T>
inline bool is_constant(const T& v)
{
    return (is_boolean(v) || is_contin(v) || is_action_result(v));
}

/// Return true if v is part of a contin-typed expression.
/// i.e. its contin istself, or is an arithmetic operator.
/// (an operator returning contin)
//
template<typename T>
inline bool is_contin_expr(const T& v)
{
    return (is_contin(v) ||
            (v == id::div) ||
            (v == id::exp) ||
            (v == id::impulse) ||
            (v == id::log) ||
            (v == id::plus) ||
            (v == id::rand) ||
            (v == id::sin) ||
            (v == id::times));
}
/// Return true if the argument is a predicate node, or negated
/// predicate node.
///
/// At this time, the only predicate in combo is greater_than_zero.
//
inline bool is_predicate(const combo_tree::iterator& it)
{
    if (*it == id::greater_than_zero) return true;
    if ((*it == id::logical_not) &&
        (*it.begin() == id::greater_than_zero)) return true;
    return false;
}

//copy a combo_tree, ignoring subtrees rooted in null vertices
void copy_without_null_vertices(combo_tree::iterator src,
                                combo_tree& dst_tr, combo_tree::iterator dst);

inline bool may_have_side_effects(combo_tree::iterator /*it*/)
{
    //TODO
    return false;
}

} // ~namespace combo
} // ~namespace opencog

// this is to be able to use std::unordered_map and such
namespace std
{
    template<>
    struct hash<opencog::combo::vertex>
    {
        size_t operator()(opencog::combo::vertex v) const
        {
            return opencog::combo::hash_value(v);
        }
    };
}

#endif
