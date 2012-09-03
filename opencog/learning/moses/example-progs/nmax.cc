/*
 * opencog/learning/moses/example-progs/nmax.cc
 *
 * Copyright (C) 2002-2008 Novamente LLC
 * All Rights Reserved
 *
 * Written by Predrag Janicic
 * Documented by Linas Vepstas, 2011
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

#include "headers.h"

using boost::lexical_cast;

// Demonstration program for the "nmax" optimization problem.  This
// is a standard learning/optimization demonstraton problem: a scoring
// function is given that totals up the values of a set of discrete
// variables.  This is the "n_max" scoring function.  The optimizer is
// supposed to be able to find the best solution to this function:
// namely, a set of variables, the value of each of which is at maximum.
// This is a generalization of the "onemax" problem, which is nmax with n=2.
//
// XXX setting n=2 currently fails due to a bug, see
// https://bugs.launchpad.net/opencog/+bug/908230
//
// NOTE: This is NOT a demonstration of program learning, which is what
// MOSES is designed for; rather, this a demonstration of the use of a
// certain component within MOSES, the so-called "optimizer". MOSES itself
// relies heavily on this optimizer to implement its meta-optimization
// algorithm.
//
// As such problems go, this is among the simplest to solve.  The code
// below illustrates how to do this using the MOSES infrastructure.
//
// This program requires five arguments:
// -- an initial seed value for the random number generator
// -- the number of discrete variables
// -- the population size
// -- the maximum number of generations to run.
// -- the number of values that variables may take.
//
// Suggest values of 0 8 70 100 5 for these.
//
// The population size should always be at least (n-1)*num_variables
// as otherwise, the least-fit individuals will be bred out of
// the population before the fittest individual is found, causing the
// algo loop forever (well, only up to max generations). There's even
// some risk of this when the population is only a little larger than
// this lower bound; maybe one-n-a-half or twice this lower bound would
// be a good choice.
//
// As output, this will print the fittest individual found at each
// generation. At the conclusion, the entire population will be printed.

int main(int argc, char** argv)
{
    // Tell the system logger to print detailed debugging messages to
    // stdout. This will let us watch what the optimizer is doing.
    // Set to Logger::WARN to show only warnings and errors.
    logger() = Logger("demo.log");
    logger().setLevel(Logger::FINE);
    logger().setPrintToStdoutFlag(true);

    // We also need to declare a specific logger for the aglo.
    // This one uses the same system logger() above, and writes all
    // messages ad the "debug" level. This allows the main loop of the
    // algo to be traced.
    cout_log_best_and_gen mlogger;

    // Parse program arguments
    vector<string> add_args{"<number of values>"};
    optargs args(argc, argv, add_args);
    int n = lexical_cast<int>(argv[5]);

    // Initialize random number generator (from the first argument
    // given to the program).
    randGen().seed(args.rand_seed);

    // Create a set of "fields". Each field is a discrete variable,
    // with 'n' different possible settings. That is, each field has
    // a multiplicity or "arity" of 'n'.  The number of such discrete
    // variables to create was passed as the second argument to the
    // program.
    //
    // For the onemax problem, 'n' would be 2.
    field_set fs(field_set::disc_spec(n), args.length);

    // Create a population of instances (bit-strings) corresponding
    // to the field specification above. The length of the bit string
    // will be ciel(log_2(n)) times the length of the field specification.
    // This is because 'n' values require at least log_2(n) bits to be
    // represented in binary.  The population size was passed as the
    // third argument to the program.
    instance_set<int> population(args.popsize, fs);

    // Initialize each member of the population to a random value.
    foreach(instance& inst, population)
        generate(fs.begin_disc(inst), fs.end_disc(inst),
                 bind(&RandGen::randint, boost::ref(randGen()), n));

    // Run the optimizer.
    // For this problem, there is no dependency at all between different
    // fields ("genes") in the field set.  Thus, for the "structure
    // learning" step, use the univariate() model, which is basically a
    // no-op; it doesn't try to learn any structure.
    //
    // The "num to select" argument is number of individuals to select
    // for learning the population distribution. For this problem, it
    // makes sense to select them all.  For smaller selections, the
    // SelectionPolicy is used to make the selection; here, the
    // tournament_selection() policy is used to select the fittest
    // individuals from the population. Since we select all, holding
    // a tournament is pointless.
    //
    // The "num to generate" is the number of individuals to create for
    // the next generation.  These are created with reference to the
    // learned model.  If the model is working well, then the created
    // individuals should be fairly fit.  In this example, it makes
    // sense to replace half the population each generation.
    // The generated individuals are then folded into the population
    // using the replace_the_worst() replacement policy. This
    // replacement policy is unconditional: the worst part of the
    // current population is replaced by the new individuals (even if
    // the new individuals are less fit than the current population!
    // But this is good enough for this example...)
    //
    // The n_max() scoring function simply totals up the values of all
    // the fields in the instance. It is defined in scoring_functions.h
    // The termination policy will halt iteration if an individual is
    // discovered to have a score of "(n-1)*args.length" -- but of
    // course, since each field takes a value from 0 to (n-1) so the
    // largest score would be the max value times the number of fields.
    //
    int num_score_evals =
    optimize(population,   // population of instances, from above.
             args.popsize,                       // num to select
             args.popsize / 2,                   // num to generate
             args.max_gens,                      // max number of generations to run
             n_max(fs),                          // ScoringPolicy
             terminate_if_gte<int>((n-1)*args.length), // TerminationPolicy
             tournament_selection(2),            // SelectionPolicy
             univariate(),                       // StructureLearningPolicy
             local_structure_probs_learning(),   // ProbsLearningPolicy
             replace_the_worst(),                // ReplacementPolicy
             mlogger);

    // The logger is asynchronous, so flush it's output before
    // writing to cout, else output will be garbled.
    logger().flush();

    cout << "A total of " << num_score_evals
         << " scoring funtion evaluations were done." << endl;

    // Show the final population
    // cout << "Final population:\n" << population << endl;
    cout << "The final population was:" << endl;
    instance_set<int>::const_iterator it = population.begin();
    for(; it != population.end(); it++) {
       cout << "Score: " << it->second
            << "\tindividual: " << population.fields().stream(it->first)
            << endl;
    }
}
