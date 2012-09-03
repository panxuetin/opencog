#include <iostream>

#include <opencog/util/mt19937ar.h>
#include <opencog/util/Logger.h>
#include <opencog/comboreduct/combo/eval.h>

#include "../representation/representation.h"
#include "../moses/moses.h"
#include "../optimization/optimization.h"
#include "../moses/scoring_functions.h"
#include "../moses/scoring.h"
#include "../moses/ann_scoring.h"


using namespace std;
using namespace boost;
using namespace opencog;
using namespace moses;
using namespace reduct;


int main(int argc, char** argv)
{

    //set flag to print only cassert and other ERROR level logs on stdout
    opencog::logger().setPrintErrorLevelStdout();
    //read maximum evaluations and RNG seed from command line
    int max_evals;
    int seed;
    bool reduce=true;
    try {
       // if (argc!=3)
        //    throw "foo";
        max_evals=lexical_cast<int>(argv[1]);
        seed=lexical_cast<int>(argv[2]);
        set_stepsize(1.25); //lexical_cast<double>(argv[3]));
        set_expansion(1.5); //lexical_cast<double>(argv[4]));
        set_depth(4) ; //exical_cast<int>(argv[5]));
        reduce = lexical_cast<int>(argv[3]);
    } catch (...) {
        cerr << "usage: " << argv[0] << " maxevals seed" << endl;
        exit(1);
    }
    
    //read in seed tree
    combo_tree tr;
    cin >> tr; 

    randGen().seed(seed);

    type_tree tt(id::lambda_type);
    tt.append_children(tt.begin(), id::ann_type, 1);

    //DOUBLE MARKOVIAN POLE TASK`
    const reduct::rule* si = &(ann_reduction());
    if(!reduce)
        si = &(clean_reduction());
    
    ann_pole2nv_score p2_score;
    ann_pole2nv_bscore p2_bscore; 
    univariate_optimization univ;
    metapopulation<ann_pole2nv_score, ann_pole2nv_bscore, univariate_optimization>
        metapop_pole2(tr, tt, *si, p2_score, p2_bscore, univ);
    moses_parameters moses_param(max_evals);
    moses::moses(metapop_pole2, moses_param);

    //change best combo tree back into ANN
    tree_transform trans; 
    combo_tree best = metapop_pole2.best_tree();
    ann bestnet = trans.decodify_tree(best);
    
    //show best network
    cout << "Best network: " << endl;
    cout << &bestnet << endl;
    //write out in dot format
    bestnet.write_dot("best_nn.dot"); 

    CartPole *the_cart;
    the_cart = new CartPole(true,false);
    the_cart->nmarkov_long=true;
    the_cart->generalization_test=false;
    double fitness = the_cart->evalNet(&bestnet);
    delete the_cart; 
    //for parameter sweep
    cout << fitness << endl;
}




