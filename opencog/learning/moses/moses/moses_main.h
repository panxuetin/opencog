/** moses_main.h ---
 *
 * Copyright (C) 2010 OpenCog Foundation
 *
 * Author: Nil Geisweiller <ngeiswei@gmail.com>
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

#ifndef _OPENCOG_MOSES_MAIN_H
#define _OPENCOG_MOSES_MAIN_H

#include <iostream>
#include <fstream>
#include <memory>
#include <stdio.h>

#include <opencog/util/mt19937ar.h>
#include <opencog/util/Logger.h>
#include <opencog/util/lru_cache.h>
#include <opencog/util/algorithm.h>
#include <opencog/util/iostreamContainer.h>
#include <opencog/util/oc_omp.h>

#include <opencog/comboreduct/combo/combo.h>

#include "moses.h"
#include "distributed_moses.h"
#include "metapopulation.h"
#include "scoring.h"

namespace opencog { namespace moses {

using namespace std;

/**
 * Run moses
 * (This could be moved to moses.h, except for a header-include order
 * clash with distributed_moses.)
 */
template<typename Score, typename BScore, typename Optimization>
void run_moses(metapopulation<Score, BScore, Optimization> &metapop,
                             const moses_parameters& moses_params)
{
    // Run moses, either on localhost, or distributed.
    if (moses_params.local)
        moses::moses(metapop, moses_params);
    else
        moses::distributed_moses(metapop, moses_params);
}

/// Print metapopulation results to stdout, logfile, etc.
struct metapop_printer
{
    metapop_printer(long _result_count,
                    bool _output_score,
                    bool _output_complexity,
                    bool _output_bscore,
                    bool _output_dominated,
                    bool _output_eval_number,
                    bool _output_with_labels,
                    const vector<string>& _labels,
                    const string& _output_file,
                    bool _output_python) :
        result_count(_result_count), output_score(_output_score),
        output_complexity(_output_complexity),
        output_bscore(_output_bscore),
        output_dominated(_output_dominated),
        output_eval_number(_output_eval_number),
        output_with_labels(_output_with_labels),
        labels(_labels),
        output_file(_output_file),
        output_python(_output_python) {}

    /**
     * Print metapopulation summary.
     */
    template<typename Score, typename BScore, typename Optimization>
    void operator()(metapopulation<Score, BScore, Optimization> &metapop) const
    {
        stringstream ss;
        metapop.ostream(ss,
                        result_count,
                        output_score,
                        output_complexity,
                        output_bscore,
                        output_dominated,
                        output_python); 
    
        if (output_eval_number)
            ss << number_of_evals_str << ": " << metapop.n_evals() << std::endl;;

#if 0
// XXX TEMPORARY HACK for perf measurements, remove me after June 2012
        extern std::vector<CTable> gtables;

        enum_table_bscore etb(gtables[0]);
        bscore_based_cscore<enum_table_bscore> straight(etb);

        enum_graded_bscore etg(gtables[0]);
        bscore_based_cscore<enum_graded_bscore> regrade(etg);

        enum_effective_bscore etf(gtables[0]);
        bscore_based_cscore<enum_effective_bscore> effect(etf);

        // const bscored_combo_tree& candidate = begin();
        auto cit = metapop.begin();
    
        for (int i=0; i<10; i++) {
            if (cit == metapop.end()) break;
            composite_score sc = straight(get_tree(*cit));
            ss << "Sraight: " << sc 
               << "\n Graded: " << get_composite_score(*cit);

            ss << "\nRegrade: " << regrade(get_tree(*cit));
            ss << "\nEffecti: " << effect(get_tree(*cit));
            ss << std::endl;
            cit++;
        }
#endif

        string res = (output_with_labels && !labels.empty()?
                      ph2l(ss.str(), labels) : ss.str());
        if (output_file.empty())
            std::cout << res;
        else {
            ofstream of(output_file.c_str());
            of << res;
            of.close();
        }
    
        // Log the best candidate
        stringstream ssb;
        metapop.ostream(ssb, 1, true, true);
        string resb = (output_with_labels && !labels.empty()?
                      ph2l(ssb.str(), labels) : ssb.str());
        if (resb.empty())
            logger().info("No candidate is good enough to be returned. Yeah that's bad!");
        else
            logger().info("Best candidates (preceded by its score):\n %s", res.c_str());
    
    #ifdef GATHER_STATS
        metapop.optimize.hiscore /= metapop.optimize.hicount;
        for (unsigned i=0; i< metapop.optimize.scores.size(); i++) {
            metapop.optimize.scores[i] /= metapop.optimize.counts[i];
            logger().info() << "Avg Scores: "
                << i << "\t"
                << metapop.optimize.hiscore << "\t"
                << metapop.optimize.counts[i] << "\t"
                << metapop.optimize.scores[i];
        }
    #endif
    }
private:
    long result_count;
    bool output_score;
    bool output_complexity;
    bool output_bscore;
    bool output_dominated;
    bool output_eval_number;
    bool output_with_labels;
    const vector<string>& labels;
    string output_file;
    bool output_python;
};


/**
 * Create metapopulation, run moses, print results.
 */
template<typename Scorer, typename BScorer, typename Printer>
void metapop_moses_results_b(const std::vector<combo_tree>& bases,
                             const opencog::combo::type_tree& tt,
                             const reduct::rule& si_ca,
                             const reduct::rule& si_kb,
                             const Scorer& sc,
                             const BScorer& bsc,
                             const optim_parameters& opt_params,
                             const metapop_parameters& meta_params,
                             const moses_parameters& moses_params,
                             Printer& printer)
{
    if (opt_params.opt_algo == hc) { // exhaustive neighborhood search
        hill_climbing climber(opt_params);

        metapopulation<Scorer, BScorer, hill_climbing>
            metapop(bases, tt, si_ca, si_kb, sc, bsc, climber, meta_params);

        run_moses(metapop, moses_params);
        printer(metapop);
    }
    else if (opt_params.opt_algo == sa) { // simulated annealing
        simulated_annealing annealer(opt_params);

        metapopulation<Scorer, BScorer, simulated_annealing>
            metapop(bases, tt, si_ca, si_kb, sc, bsc, annealer, meta_params);

        run_moses(metapop, moses_params);
        printer(metapop);
    }
    else if (opt_params.opt_algo == un) { // univariate
        univariate_optimization unopt(opt_params);

        metapopulation<Scorer, BScorer, univariate_optimization>
            metapop(bases, tt, si_ca, si_kb, sc, bsc, unopt, meta_params);

        run_moses(metapop, moses_params);
        printer(metapop);
    }
    else {
        std::cerr << "Unknown optimization algo " << opt_params.opt_algo
                  << ". Supported algorithms are un (for univariate),"
                  << " sa (for star-shaped search) and hc (for local search)"
                  << std::endl;
        exit(1);
    }
}

/**
 * like above, but assumes that the score is bscore based
 */
template<typename BScorer, typename Printer>
void metapop_moses_results(const std::vector<combo_tree>& bases,
                           const opencog::combo::type_tree& type_sig,
                           const reduct::rule& si_ca,
                           const reduct::rule& si_kb,
                           BScorer& bscorer,
                           optim_parameters opt_params,
                           const metapop_parameters& meta_params,
                           moses_parameters moses_params,
                           Printer& printer)
{
    bscore_based_cscore<BScorer> c_scorer(bscorer);

    // update terminate_if_gte and max_score criteria
    score_t bps = c_scorer.best_possible_score();
    score_t target_score = std::min(moses_params.max_score, bps);
    opt_params.terminate_if_gte = target_score;
    // update minimum score improvement
    opt_params.set_min_score_improv(c_scorer.min_improv());
    moses_params.max_score = target_score;
    logger().info("Target score = %f", target_score);

    if (meta_params.enable_cache) {
        static const unsigned initial_cache_size = 1000000;
        
        if (meta_params.include_dominated) {
            // When the include_dominated flag is set, then trees are merged
            // into the metapop based only on the score (and complexity),
            // not on the behavioral score. So we can throw away the 
            // behavioral score after computng it (we don't need to cche it).
            typedef bscore_based_cscore<BScorer> CScorer;
            typedef adaptive_cache<prr_cache_threaded<CScorer> > ScoreACache;
            CScorer cscorer(bscorer);
            prr_cache_threaded<CScorer> score_cache(initial_cache_size, cscorer);
            ScoreACache score_acache(score_cache, "composite scores");
            metapop_moses_results_b(bases, type_sig, si_ca, si_kb,
                                    score_acache, bscorer,
                                    opt_params, meta_params, moses_params,
                                    printer);
        }
        else {
            // We put the cache on the bscore as well because then it
            // is reused later (for metapopulation merging)
            typedef prr_cache_threaded<BScorer> BScoreCache;
            typedef adaptive_cache<BScoreCache> BScoreACache;
            typedef bscore_based_cscore<BScoreACache> CScorer;
            BScoreCache bscore_cache(initial_cache_size, bscorer);
            BScoreACache bscore_acache(bscore_cache, "behavioural scores");
            typedef adaptive_cache<prr_cache_threaded<CScorer> > ScoreACache;
            CScorer cscorer(bscore_acache);
            prr_cache_threaded<CScorer> score_cache(initial_cache_size, cscorer);
            ScoreACache score_acache(score_cache, "composite scores");
            metapop_moses_results_b(bases, type_sig, si_ca, si_kb,
                                    score_acache, bscore_acache,
                                    opt_params, meta_params, moses_params,
                                    printer);
        }
        return;
    }

    metapop_moses_results_b(bases, type_sig, si_ca, si_kb, c_scorer, bscorer,
                         opt_params, meta_params, moses_params, printer);
}

} // ~namespace moses
} // ~namespace opencog

#endif // _OPENCOG_MOSES_MAIN_H
