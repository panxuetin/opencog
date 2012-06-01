/** feature-selection.cc ---
 *
 * Copyright (C) 2011 OpenCog Foundation
 *
 * Author: Nil Geisweiller <nilg@desktop>
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

#include <iostream>
#include <fstream>
#include <memory>
#include <stdio.h>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <opencog/util/mt19937ar.h>
#include <opencog/util/Logger.h>
#include <opencog/util/lru_cache.h>
#include <opencog/util/algorithm.h>
#include <opencog/util/iostreamContainer.h>
#include <opencog/util/log_prog_name.h>

#include <opencog/comboreduct/combo/table.h>

#include <opencog/learning/moses/optimization/optimization.h>

#include "feature-selection.h"
#include "../feature_optimization.h"
#include "../feature_scorer.h"

using namespace std;
using namespace opencog;
using namespace combo;

using namespace boost::program_options;
using boost::lexical_cast;
using boost::trim;

// Assorted defaults.
const static unsigned max_filename_size = 255;

static const string default_log_file_prefix = "feature-selection";
static const string default_log_file_suffix = "log";
static const string default_log_file = default_log_file_prefix + "." + default_log_file_suffix;

// Program option names and abbreviations.
static const pair<string, string> rand_seed_opt("random-seed", "r");
static const pair<string, string> algo_opt("algo", "a");
static const pair<string, string> input_data_file_opt("input-file", "i");
static const pair<string, string> target_feature_opt("target-feature", "u");
static const pair<string, string> ignore_feature_str_opt("ignore-feature", "Y");
static const pair<string, string> max_evals_opt("max-evals", "m");
static const pair<string, string> output_file_opt("output-file", "o");
static const pair<string, string> log_level_opt("log-level", "l");
static const pair<string, string> log_file_opt("log-file", "F");
static const pair<string, string> log_file_dep_opt_opt("log-file-dep-opt", "L");
static const pair<string, string> target_size_opt("target-size", "C");
static const pair<string, string> threshold_opt("threshold", "T");
static const pair<string, string> jobs_opt("jobs", "j");
static const pair<string, string> inc_target_size_epsilon_opt("inc-target-size-epsilon", "E");
static const pair<string, string> inc_redundant_intensity_opt("inc-redundant-intensity", "D");
static const pair<string, string> inc_interaction_terms_opt("inc-interaction-terms", "U");
static const pair<string, string> hc_initial_feature_opt("initial-feature", "f");
static const pair<string, string> hc_max_score_opt("max-score", "A");
static const pair<string, string> hc_confidence_penalty_intensity_opt("confidence-penalty-intensity", "c");
static const pair<string, string> hc_fraction_of_remaining_opt("hc-fraction-of-remaining", "O");
static const pair<string, string> hc_cache_size_opt("cache-size", "s");

// Returns a string interpretable by Boost.Program_options
// "name,abbreviation"
string opt_desc_str(const pair<string, string>& opt) {
    string res = string(opt.first);
    if (!opt.second.empty())
        res += string(",") + opt.second;
    return res;
}

/**
 * Display error message about unsupported type and exit
 */
void unsupported_type_exit(const type_tree& tt)
{
    std::cerr << "error: type " << tt << "currently not supported" << std::endl;
    exit(1);
}

void unsupported_type_exit(type_node type)
{
    unsupported_type_exit(type_tree(type));
}

int main(int argc, char** argv)
{
    unsigned long rand_seed;
    string log_level;
    string log_file;
    bool log_file_dep_opt;
    feature_selection_parameters fs_params;
    string target_feature_str;
    vector<string> ignore_features_str;

    // Declare the supported options.
    options_description desc("Allowed options");

    desc.add_options()
        ("help,h", "Produce help message.\n")

        (opt_desc_str(algo_opt).c_str(),
         value<string>(&fs_params.algorithm)->default_value(mmi),
         string("Feature selection algorithm. Supported algorithms are:\n")
             /*
              * We're not going to support univariate or sa any time
              * soon, and maybe never; they're kind-of deprecated in
              * MOSES, at the moment.
             .append(un).append(" for univariate,\n")
             .append(sa).append(" for simulated annealing,\n")
             */
             .append(mmi).append(" for maximal mutual information,\n")
             .append(moses::hc).append(" for hillclimbing,\n")
             .append(inc).append(" for incremental mutual information.\n").c_str())

        // ======= File I/O opts =========
        (opt_desc_str(input_data_file_opt).c_str(),
         value<string>(&fs_params.input_file),
         "Input table file in DSV format (seperators are comma, whitespace and tabulation).\n")

        (opt_desc_str(target_feature_opt).c_str(),
         value<string>(&target_feature_str),
         "Label of the target feature to fit. If none is given the first one is used.\n")

        (opt_desc_str(ignore_feature_str_opt).c_str(),
         value<vector<string>>(&ignore_features_str),
         "Ignore feature from the datasets. Can be used several times "
         "to ignore several features.\n")

        (opt_desc_str(output_file_opt).c_str(),
         value<string>(&fs_params.output_file),
         "File where to save the results. If empty then it outputs on the stdout.\n")

        (opt_desc_str(log_level_opt).c_str(),
         value<string>(&log_level)->default_value("DEBUG"),
         "Log level; verbosity of logging and debugging messages to "
         "write. Possible levels are NONE, ERROR, WARN, INFO, DEBUG, "
         "FINE. Case does not matter.\n")

        (opt_desc_str(log_file_opt).c_str(),
         value<string>(&log_file)->default_value(default_log_file),
         string("File name where to record the output log.\n")
         .c_str())

        (opt_desc_str(log_file_dep_opt_opt).c_str(),
         string("Use an option-dependent logfile name. The name of "
          "the log is determined by the command-line options; the "
          "base (prefix) of the file is that given by the -F option.  "
          "So, for instance, if the options -L foo -r 123 are given, "
          "then the logfile name will be foo_random-seed_123.log.  "
          "The filename will be truncated to a maximum of ")
          .append(lexical_cast<string>(max_filename_size))
          .append(" characters.\n").c_str())

        // ======= Generic algo opts =========
        (opt_desc_str(jobs_opt).c_str(),
         value<unsigned>(&fs_params.jobs)->default_value(1),
         string("Number of threads to use.\n").c_str())

        (opt_desc_str(target_size_opt).c_str(),
         value<unsigned>(&fs_params.target_size)->default_value(0),
            "Feature count.  This option "
            "specifies the number of features to be selected out of "
            "the dataset.  A value of 0 disables this option. \n")

        (opt_desc_str(threshold_opt).c_str(),
         value<double>(&fs_params.threshold)->default_value(0),
            "Improvment threshold. Floating point number. "
            "Specifies the threshold above which the mutual information "
            "of a feature is considered to be significantly correlated "
            "to the target.  A value of zero means that all features "
            "will be selected. \n"
            "For the -ainc algo only, the -C flag over-rides this setting.\n")

        (opt_desc_str(rand_seed_opt).c_str(),
         value<unsigned long>(&rand_seed)->default_value(1),
         "Random seed.\n")

        // ======= Incremental selection params =======
        (opt_desc_str(inc_redundant_intensity_opt).c_str(),
         value<double>(&fs_params.inc_red_intensity)->default_value(0.1),
         "Incremental Selection parameter. Floating-point value must "
         "lie between 0.0 and 1.0.  A value of 0.0 means that no "
         "redundant features will discarded, while 1.0 will cause a "
         "maximal number will be discarded.\n")

        (opt_desc_str(inc_target_size_epsilon_opt).c_str(),
         value<double>(&fs_params.inc_target_size_epsilon)->default_value(0.001),
         "Incremental Selection parameter. Tolerance applied when "
         "selecting for a fixed number of features (option -C).\n")

        (opt_desc_str(inc_interaction_terms_opt).c_str(),
         value<unsigned>(&fs_params.inc_interaction_terms)->default_value(1),
         "Incremental Selection parameter. Maximum number of "
         "interaction terms considered during incremental feature "
         "selection. Higher values make the feature selection more "
         "accurate but is combinatorially more computationally expensive.\n")

        // ======= Hill-climbing only params =======
        (opt_desc_str(hc_max_score_opt).c_str(),
         value<double>(&fs_params.hc_max_score)->default_value(1),
         "Hillclimbing parameter.  The max score to reach, once "
         "reached feature selection halts.\n")

        (opt_desc_str(hc_confidence_penalty_intensity_opt).c_str(),
         value<double>(&fs_params.hc_confi)->default_value(1.0),
         "Hillclimbing parameter.  Intensity of the confidence "
         "penalty, in the range [0,+Inf).  Zero means no confidence "
         "penalty. This parameter influences how much importance is "
         "attributed to the confidence of the quality measure. The "
         "fewer samples in the data set, the more features the "
         "less confidence in the feature set quality measure.\n")

        (opt_desc_str(hc_initial_feature_opt).c_str(), 
         value<vector<string> >(&fs_params.hc_initial_features),
         "Hillclimbing parameter.  Initial feature to search from.  "
         "This option can be used as many times as there are features, "
         "to have them included in the initial feature set. If the "
         "initial feature set is close to the one that maximizes the "
         "quality measure, the selection speed can be greatly increased.\n")

        (opt_desc_str(max_evals_opt).c_str(),
         value<unsigned>(&fs_params.max_evals)->default_value(10000),
         "Hillclimbing parameter.  Maximum number of fitness function "
         "evaluations.\n")

        (opt_desc_str(hc_fraction_of_remaining_opt).c_str(),
         value<double>(&fs_params.hc_fraction_of_remaining)->default_value(0.5),
         "Hillclimbing parameter.  Determine the fraction of the "
         "remaining number of eval to use for the current iteration.\n")

        (opt_desc_str(hc_cache_size_opt).c_str(),
         value<unsigned long>(&fs_params.hc_cache_size)->default_value(1000000),
         "Hillclimbing parameter.  Cache size, so that identical "
         "candidates are not re-evaluated.   Zero means no cache.\n")

        ;

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    // Set flags
    log_file_dep_opt = vm.count(log_file_dep_opt_opt.first) > 0;

    // Help
    if (vm.count("help") || argc == 1) {
        cout << desc << std::endl;
        return 1;
    }

    // Set log
    if (log_file_dep_opt) {
        std::set<std::string> ignore_opt {
            log_file_dep_opt_opt.first,
            log_file_opt.first
        };

        // If the user specified a log file with -F then treat this
        // as the prefix to the long filename.
        string log_file_prefix = default_log_file_prefix;
        if (log_file != default_log_file) {
            log_file_prefix = log_file;
        }

        log_file = determine_log_name(log_file_prefix,
                                      vm, ignore_opt,
                    std::string(".").append(default_log_file_suffix));
    }

    // Remove any existing log files.
    remove(log_file.c_str());
    logger().setFilename(log_file);
    trim(log_level);
    Logger::Level level = logger().getLevelFromString(log_level);
    if (level == Logger::BAD_LEVEL) {
        cerr << "Fatal Error: Log level " << log_level
             << " is incorrect (see --help)." << endl;
        exit(1);
    }
    logger().setLevel(level);
    logger().setBackTraceLevel(Logger::ERROR);

    // Log command-line args
    string cmdline = "Command line:";
    for (int i = 0; i < argc; ++i) {
         cmdline += " ";
         cmdline += argv[i];
    }
    logger().info(cmdline);
    
    // init random generator
    randGen().seed(rand_seed);

    // setting OpenMP parameters
    setting_omp(fs_params.jobs);

    // Logger
    logger().info("Read input file %s", fs_params.input_file.c_str());
    // ~Logger

    // Find the position of the target feature (the first one by default)
    fs_params.target_feature = target_feature_str.empty()? 0
        : find_feature_position(fs_params.input_file, target_feature_str);

    // Get the list of indexes of features to ignore
    vector<int> ignore_features;
    fs_params.ignore_features = find_features_positions(fs_params.input_file,
                                                        ignore_features_str);
    ostreamContainer(logger().info() << "Ignore the following columns: ",
                     fs_params.ignore_features);

    OC_ASSERT(boost::find(fs_params.ignore_features, fs_params.target_feature)
              == fs_params.ignore_features.end(),
              "You cannot ignore the target feature (column %d)",
              fs_params.target_feature);
    
    // Read input_data_file file
    Table table = loadTable(fs_params.input_file,
                            fs_params.target_feature,
                            fs_params.ignore_features);

    type_tree inferred_tt = infer_data_type_tree(fs_params.input_file,
                                                 fs_params.target_feature,
                                                 fs_params.ignore_features);
    type_tree output_tt = get_signature_output(inferred_tt);
    type_node inferred_type = get_type_node(output_tt);

    // Go and do it.
    if(inferred_type == id::boolean_type) {
        feature_selection(table, fs_params);
    } else {
        unsupported_type_exit(inferred_type);
    }
}
