/** ContextualizerRule.cc --- 
 *
 * Copyright (C) 2010 OpenCog Foundation
 *
 * Author: Nil Geisweiller <nilg@laptop>
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

#include "ContextualizerRule.h"
#include <opencog/util/algorithm.h>
#include "../RuleFunctions.h"

namespace opencog { namespace pln {

meta ContextualizerRule::i2oType(const VertexSeq& h) const {
    OC_ASSERT(h.size() == 1);
    pHandle ph = _v2h(h[0]);

    OC_ASSERT(asw->getArity(ph) == 2); // maybe this should be relaxed
    pHandle A1 = asw->getOutgoing(ph,0);
    pHandle A2 = asw->getOutgoing(ph,1);

    // case a) contextualizing a relation
    if(asw->getType(A1) == AND_LINK && asw->getType(A2) == AND_LINK) {
        // @todo generilize for n-ary
        OC_ASSERT(asw->getArity(A1) == 2);
        OC_ASSERT(asw->getArity(A2) == 2);
        pHandleSeq out1 = asw->getOutgoing(A1);
        pHandleSeq out2 = asw->getOutgoing(A2);

        // the intersection btw out1 and out2 will be the new context
        pHandleSeq inter = set_intersection(out1, out2);
        if(inter.size() > 0) {
            pHandle CX = inter[0];
            // can be optimized
            pHandleSeq new_out1 = set_difference(out1, inter);
            pHandleSeq new_out2 = set_difference(out2, inter);
            OC_ASSERT(new_out1.size() == 1); // @todo generilize for n-ary
            pHandle A = new_out1[0];
            OC_ASSERT(new_out2.size() == 1); // @todo generilize for n-ary
            pHandle B = new_out2[0];
            return meta(new vtree(mva((pHandle)CONTEXT_LINK, mva(CX),
                                      mva((pHandle)asw->getType(ph),
                                          mva(A), mva(B)))));
        }
    }

    // case b) and c)
    if(asw->getType(ph) == SUBSET_LINK) {
        if(asw->isSubType(A2, SATISFYING_SET_LINK)) // case c)
            A2 = asw->getOutgoing(A2, 0);
        return meta(new vtree(mva((pHandle)CONTEXT_LINK, mva(A1), mva(A2))));
    }

    OC_ASSERT(false); // unhandle case
    return meta();
}

TVSeq ContextualizerRule::formatTVarray(const VertexSeq& premiseArray) const {
    OC_ASSERT(premiseArray.size()==1);
    return TVSeq(1, asw->getTV(_v2h(premiseArray[0])));
}

ContextualizerRule::ContextualizerRule(AtomSpaceWrapper* _asw)
    : super(_asw, false, "ContextualizerRule") {
    inputFilter.push_back(meta(new tree<Vertex>(mva((pHandle)ATOM))));
    /// @todo understand why the following does work
    // inputFilter.push_back(meta(new tree<Vertex>(mva((pHandle)LINK,
    //                                                 mva((pHandle)ATOM),
    //                                                 mva((pHandle)ATOM)))));
}

Rule::setOfMPs ContextualizerRule::o2iMetaExtra(meta outh,
                                                bool& overrideInputFilter) const {
    vtree::iterator root = outh->begin();
    pHandle ph = _v2h(*root);

    if(!asw->isSubType(ph, CONTEXT_LINK))
        return Rule::setOfMPs();

    // @todo it should probably be better to first try to
    // contextualize a relation and then only if it's not possible try
    // the Node contextualization

    OC_ASSERT(root.number_of_children() == 2);
    vtree::iterator CX_it = root.begin();
    pHandle CX = _v2h(*CX_it);
    vtree::iterator A_it = root.last_child();
    pHandle A = _v2h(*A_it);

    BoundVTree* res;
    // case a) contextualizing a relation
    if(asw->isSubType(A, LINK) || !asw->isSubType(A, SATISFYING_SET_LINK)) {
        OC_ASSERT(A_it.number_of_children() == 2); // @todo generalize for n-ary
        res = new BoundVTree(mva((pHandle)asw->getType(A),
                                 mva((pHandle)AND_LINK, mva(CX),
                                     vtree(A_it.begin())),
                                 mva((pHandle)AND_LINK, mva(CX),
                                     vtree(A_it.last_child()))));
    }
    // case b) and c)
    // @note it is missing the cases where concepts are AND_LINK and such
    else {
        if(asw->isSubType(A, SATISFYING_SET_LINK)) // case c)
            A = _v2h(*A_it.begin());
        res = new BoundVTree(mva((pHandle)SUBSET_LINK, mva(CX), mva(A)));
    }

    overrideInputFilter = true;

    return makeSingletonSet(Rule::MPs(1, BBvtree(res)));
}
        
meta ContextualizerRule::targetTemplate() const {
    return(meta(new vtree(mva((pHandle)CONTEXT_LINK, 
                              vtree(CreateVar(asw)),
                              vtree(CreateVar(asw))))));
}

}} // namespace opencog { namespace pln {
