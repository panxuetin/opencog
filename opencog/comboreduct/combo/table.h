/** table.h ---
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


#ifndef _OPENCOG_TABLE_H
#define _OPENCOG_TABLE_H

#include <fstream>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/algorithm/find.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/tokenizer.hpp>

#include <opencog/util/RandGen.h>
#include <opencog/util/iostreamContainer.h>
#include <opencog/util/dorepeat.h>
#include <opencog/util/Counter.h>

#include "eval.h"
#include "vertex.h"
#include "common_def.h"

#define COEF_SAMPLE_COUNT 20.0 // involved in the formula that counts
                               // the number of trials needed to check
                               // a formula

namespace opencog { namespace combo {

using boost::variant;
using boost::adaptors::map_values;

///////////////////
// Generic table //
///////////////////

static const std::string default_input_label("i");

/// CTable is a "compressed" table.  Compression is done by removing
/// duplicated inputs, and the output column is replaced by a counter
/// of the duplicated outputs.  That is, the output column is of the
/// form {v1:c1, v2:c2, ...} where c1 is the number of times value v1
/// was seen in the output, c2 the number of times v2 was observed, etc.
///
/// For example, if one has the following table:
///
///   output, input1, input2
///   1,1,0
///   0,1,1
///   1,1,0
///   0,1,0
///
/// Then the compressed table is
///
///   output, input1, input2
///   {0:1,1:2},1,0
///   {0:1},1,1
///
/// Most scoring functions work on CTable, as it avoids re-evaluating a
/// combo program on the same inputs.
//
class CTable : public std::map<vertex_seq, Counter<vertex, unsigned>>
{
public:
    typedef vertex_seq key_type;
    typedef Counter<vertex, unsigned> counter_t;
    typedef std::map<key_type, counter_t> super;
    typedef typename super::value_type value_type;

    std::string olabel;               // output label
    std::vector<std::string> ilabels; // list of input labels

    // definition is delaied after Table as it uses Table
    template<typename Func>
    CTable(const Func& func, arity_t arity, int nsamples = -1);
    
    CTable(const std::string& _olabel, const std::vector<std::string>& _ilabels)
        : olabel(_olabel), ilabels(_ilabels) {}


    // TODO remove that junk  !?? Remove what junk ?? Huh ?? XXX
    binding_map get_binding_map(const vertex_seq& args) const
    {
        binding_map bmap;
        for (size_t i = 0; i < args.size(); ++i)
            bmap[i+1] = args[i];
        return bmap;
    }

    // like above but only consider the arguments in as
    binding_map get_binding_map(const vertex_seq& args, const arity_set& as) const
    {
        binding_map bmap;
        foreach (arity_t a, as)
            bmap[a] = args[a-1];
        return bmap;
    }

    // Return the total number of observations (should be equal to the
    // size of the corresponding uncompressed table)
    unsigned uncompressed_size() const {
        unsigned res = 0;
        foreach(const value_type& v, *this) {
            res += v.second.total_count();
        }
        return res;
    }

    type_tree tt;
};

/**
 * Input table of vertexes.
 * Rows represent data samples.
 * Columns represent input variables.
 * Optionally holds a list of column labels (input variable names)
 */
class ITable : public std::vector<vertex_seq>
{
public:
    typedef std::vector<vertex_seq > super;
    ITable();
    ITable(const super& mat,
           std::vector<std::string> il = std::vector<std::string>());
    /**
     * generate an input table according to the signature tt.
     *
     * @param tt signature of the table to generate.
     * @param nsamples sample size, if negative then the sample
              size is automatically determined.
     * @param min_contin minimum contin value.
     * @param max_contin maximum contin value.
     *
     * It onyl works for contin-boolean signatures
     */
    // min_contin and max_contin are used in case tt has contin inputs
    ITable(const type_tree& tt, int nsamples = -1,
           contin_t min_contin = -1.0, contin_t max_contin = 1.0);

    // set input labels
    void set_labels(const std::vector<std::string>& il);
    const std::vector<std::string>& get_labels() const;

    // like get_labels but filtered accordingly to a container of
    // arity_t. Each value of that container corresponds to the column
    // index of the ITable (starting from 0).
    template<typename F>
    std::vector<std::string> get_filtered_labels(const F& filter)
    {
        std::vector<std::string> res;
        foreach(arity_t a, filter)
            res.push_back(get_labels()[a]);
        return res;
    }

    // get binding map prior calling the combo evaluation
    binding_map get_binding_map(const vertex_seq& args) const
    {
        binding_map bmap;
        for(size_t i = 0; i < args.size(); ++i)
            bmap[i+1] = args[i];
        return bmap;
    }
    // like above but only consider the arguments in as
    binding_map get_binding_map(const vertex_seq& args, const arity_set& as) const
    {
        binding_map bmap;
        foreach (arity_t a, as)
            bmap[a] = args[a-1];
        return bmap;
    }

    arity_t get_arity() const {
        return super::front().size();
    }

    bool operator==(const ITable& rhs) const
    {
        return
            static_cast<const super&>(*this) == static_cast<const super&>(rhs)
            && get_labels() == rhs.get_labels();
    }

    /// return a copy of the input table filtered according to a given
    /// container of arity_t. Each value of that container corresponds
    /// to the column index of the ITable (starting from 0).
    template<typename F>
    ITable filter(const F& f)
    {
        ITable res;
        res.set_labels(get_filtered_labels(f));
        foreach(const value_type& row, *this) {
            vertex_seq new_row;
            foreach(arity_t a, f)
                new_row.push_back(row[a]);
            res.push_back(new_row);
        }
        return res;
    }

protected:
    mutable std::vector<std::string> labels; // list of input labels

private:
    std::vector<std::string> get_default_labels() const
    {
        std::vector<std::string> res;
        for(arity_t i = 1; i <= get_arity(); ++i)
            res.push_back(default_input_label
                          + boost::lexical_cast<std::string>(i));
        return res;
    }

    /**
     * this function take an arity in input and returns in output the
     * number of samples that would be appropriate to check the semantics
     * of its associated tree.
     *
     * Note : could take the two trees to checking and according to their
     * arity structure, whatever, find an appropriate number.
     */
    unsigned sample_count(arity_t contin_arity)
    {
        if (contin_arity == 0)
            return 1;
        else return COEF_SAMPLE_COUNT*log(contin_arity + EXPONENTIAL);
    }

};

static const std::string default_output_label("output");

/**
 * Output table of vertexes.
 * Rows represent dependent data samples.
 * There is only one column: a single output value for each row.
 * Optionally holds a column label (output variable names)
 */
class OTable : public vertex_seq
{
    typedef vertex_seq super;
public:
    typedef vertex value_type;

    OTable(const std::string& ol = default_output_label);
    OTable(const super& ot, const std::string& ol = default_output_label);

    /// Construct the OTable by evaluating the combo tree @tr for each
    /// row in the input ITable.
    OTable(const combo_tree& tr, const ITable& itable,
           const std::string& ol = default_output_label);

    /// Construct the OTable by evaluating the combo tree @tr for each
    /// row in the input CTable.
    OTable(const combo_tree& tr, const CTable& ctable,
           const std::string& ol = default_output_label);

    template<typename Func>
    OTable(const Func& f, const ITable& it,
           const std::string& ol = default_output_label)
        : label(ol)
    {
        foreach(const vertex_seq& vs, it)
            push_back(f(vs.begin(), vs.end()));
    }

    void set_label(const std::string& ol);
    const std::string& get_label() const;
    bool operator==(const OTable& rhs) const;
    contin_t abs_distance(const OTable& ot) const;
    contin_t sum_squared_error(const OTable& ot) const;
    contin_t mean_squared_error(const OTable& ot) const;
    contin_t root_mean_square_error(const OTable& ot) const;

    vertex get_enum_vertex(const std::string& token);

private:
    std::string label; // output label
};

/**
 * Typed data table.
 * The table consists of an ITable of inputs (independent variables),
 * an OTable holding the output (the dependent variable), and a type
 * tree identifiying the types of the inputs and outputs.
 */
struct Table
{
    typedef vertex value_type;

    Table();

    template<typename Func>
    Table(const Func& func, arity_t a, int nsamples = -1) :
        tt(gen_signature(type_node_of<bool>(),
                         type_node_of<bool>(), a)),
        itable(tt), otable(func, itable) {}
    
    Table(const combo_tree& tr, int nsamples = -1,
          contin_t min_contin = -1.0, contin_t max_contin = 1.0);
    size_t size() const { return itable.size(); }
    arity_t get_arity() const { return itable.get_arity(); }
    // Filter according to a container of arity_t. Each value of that
    // container corresponds to the column index of the ITable
    // (starting from 0).
    template<typename F> Table filter(const F& f) {
        Table res;
        res.itable = itable.filter(f);
        res.otable = otable;
        return res;
    }
    /// return the corresponding compressed table
    CTable compress() const;

    type_tree tt;
    ITable itable;
    OTable otable;
};

template<typename Func>
CTable::CTable(const Func& func, arity_t arity, int nsamples) {
    Table table(func, arity, nsamples);
    *this = table.compress();
}

        
////////////////////////
// Mutual Information //
////////////////////////

/**
 * Compute the joint entropy H(Y) of an output table. It assumes the data
 * are discretized. (?)
 */
double OTEntropy(const OTable& ot);

/**
 * Compute the mutual information between a set of independent features
 * X_1, ... X_n and a taget feature Y.
 *
 * The target (output) featuer Y is provided in the output table OTable,
 * whereas the input features are specified as a set of indexes giving
 * columns in the input table ITable.
 *
 * The mutual information 
 *
 *   MI(Y; X1, ..., Xn)
 *
 * is computed as
 *
 *   MI(Y;X1, ..., Xn) = H(X1, ..., Xn) + H(Y) - H(X1, ..., Xn, Y)
 *
 * where
 *   H(...) are the joint entropies.
 *
 * @note only works for discrete data set.
 */
template<typename FeatureSet>
double mutualInformation(const ITable& it, const OTable& ot, const FeatureSet& fs)
{
    // The following mapping is used to keep track of the number
    // of inputs a given setting. For instance, X1=false, X2=true,
    // X3=true is one possible setting. It is then used to compute
    // H(Y, X1, ..., Xn) and H(X1, ..., Xn)
    typedef Counter<vertex_seq, unsigned> VSCounter;
    VSCounter ic, // for H(X1, ..., Xn)
        ioc; // for H(Y, X1, ..., Xn)
    ITable::const_iterator i_it = it.begin();
    OTable::const_iterator o_it = ot.begin();
    for(; i_it != it.end(); ++i_it, ++o_it) {
        vertex_seq ic_vec;
        foreach(const typename FeatureSet::value_type& idx, fs)
            ic_vec.push_back((*i_it)[idx]);
        ++ic[ic_vec];
        vertex_seq ioc_vec(ic_vec);
        ioc_vec.push_back(*o_it);
        ++ioc[ioc_vec];
    }

    // Compute the probability distributions
    std::vector<double> ip(ic.size()), iop(ioc.size());
    double total = it.size();
    auto div_total = [&](unsigned c) { return c/total; };
    transform(ic | map_values, ip.begin(), div_total);
    transform(ioc | map_values, iop.begin(), div_total);

    // Compute the joint entropies
    return entropy(ip) + OTEntropy(ot) - entropy(iop);
}

// Like the above, but taking a table in argument instead of 
// input and output tables
template<typename FeatureSet>
double mutualInformation(const Table& table, const FeatureSet& fs)
{
    return mutualInformation(table.itable, table.otable, fs);
}

/**
 * Like above but uses a compressed table instead of input and output
 * table. It assumes the output is boolean. The CTable cannot be
 * passed as const because the use of the operator[] may modify it's
 * content (by adding default value on missing keys).
 */
template<typename FeatureSet>
double mutualInformation(const CTable& ctable, const FeatureSet& fs)
{
    // the following mapping is used to keep track of the number
    // of inputs a given setting. For instance X1=false, X2=true,
    // X3=true is one possible setting. It is then used to compute
    // H(Y, X1, ..., Xn) and H(X1, ..., Xn)
    typedef Counter<vertex_seq, unsigned> VSCounter;
    VSCounter ic, // for H(X1, ..., Xn)
        ioc; // for H(Y, X1, ..., Xn)
    unsigned oc = 0; // for H(Y)
    double total = 0;
    foreach(const auto& row, ctable) {
        unsigned falses = row.second.get(id::logical_false);
        unsigned trues = row.second.get(id::logical_true);
        unsigned row_total = falses + trues;
        // update ic
        vertex_seq vec;
        foreach(unsigned idx, fs)
            vec.push_back(row.first[idx]);
        ic[vec] += row_total;
        // update ioc
        if (falses > 0) {
            vec.push_back(id::logical_false);
            ioc[vec] += falses;
            vec.pop_back();
        }
        if (trues > 0) {
            vec.push_back(id::logical_true);
            ioc[vec] += trues;
        }
        // update oc
        oc += trues;
        // update total
        total += row_total;
    }
    // Compute the probability distributions
    std::vector<double> ip(ic.size()), iop(ioc.size());
    auto div_total = [&](unsigned c) { return c/total; };
    transform(ic | map_values, ip.begin(), div_total);
    transform(ioc | map_values, iop.begin(), div_total);
    // Compute the entropies
    return entropy(ip) + binaryEntropy(oc/total) - entropy(iop);
}

//////////////////
// istreamTable //
//////////////////

/**
 * remove the carriage return (for DOS format)
 */
void removeCarriageReturn(std::string& str);

/**
 * remove non ASCII char at the begining of the string
 */
void removeNonASCII(std::string& str);

/**
 * Return true if the next chars in 'in' correspond to carriage return
 * (support UNIX and DOS format) and advance in of the checked chars.
 */
bool checkCarriageReturn(std::istream& in);

/**
 * Return the arity of the table provided in istream (by counting the
 * number of elements of the first line).
 */
arity_t istreamArity(std::istream& in);
/**
 * Helper, like above but given the file name instead of istream
 */
arity_t dataFileArity(const std::string& dataFileName);

std::vector<std::string> loadHeader(const std::string& file_name);

/**
 * check if the data file has a header. That is whether the first row
 * starts with a sequence of output and input labels
 */
bool hasHeader(const std::string& dataFileName);

/**
 * Check the token, if it is "0" or "1" then it is boolean, otherwise
 * it is contin. It is not 100% reliable of course and should be
 * improved.
 */
type_node infer_type_from_token(const std::string& token);

/**
 * take a row in input as a pair {inputs, output} and return the type
 * tree corresponding to the function mapping inputs to output. If the
 * inference fails then it returns a type_tree with
 * id::ill_formed_type as root.
 */
type_tree infer_row_type_tree(std::pair<std::vector<std::string>,
                                        std::string>& row);

// used as default of infer_data_type_tree and other IO function
static const std::vector<int> empty_int_vec;

/// Create a type tree describing the types of the input columns
/// and the output column.
///        
/// @param output_col_num is the column we expect to use as the output
/// (the dependent variable)
///
/// @param ignore_col_nums are a list of column to ignore
///
/// @return type_tree infered
type_tree infer_data_type_tree(const std::string& fileName,
                               int output_col_num = 0,
                               const std::vector<int>& ignore_col_nums
                               = empty_int_vec);

/**
 * Find the column numbers associated with the names features
 * 
 * If the target begins with an alpha character, it is assumed to be a
 * column label. We return the column number; 0 is the left-most column.
 *
 * If the target is numeric, just assum that it is a column number.
 */
std::vector<int> find_features_positions(const std::string& fileName,
                                         const std::vector<std::string>& features);
int find_feature_position(const std::string& fileName,
                          const std::string& feature);

/**
 * Take a row, strip away any nnon-ASCII chars and trailing carriage
 * returns, and then return a tokenizer.  Tokenization uses the
 * seperator characters comma, blank, tab (',', ' ' or '\t').
 */
typedef boost::tokenizer<boost::escaped_list_separator<char>> table_tokenizer;
table_tokenizer get_row_tokenizer(std::string& line);

/**
 * Take a line and return a vector containing the elements parsed.
 * Used by istreamTable. This will modify the line to remove leading
 * non-ASCII characters, as well as stripping of any carriage-returns.
 */
template<typename T>
std::vector<T> tokenizeRow(std::string& line)
{
    table_tokenizer tok = get_row_tokenizer(line);
    std::vector<T> res;
    foreach (const std::string& t, tok)
        res.push_back(boost::lexical_cast<T>(t));
    return res;
}

/**
 * Take a line and return an output and a vector of inputs.
 *
 * The pos variable indicates which token is taken as the output.
 * If pos < 0 then the last token is assumed to be the output.
 * If pos >=0 then that token is used (0 is the first, 1 is the
 * second, etc.)  If pos is out of range, an assert is raised.
 *
 * This will modify the line to remove leading non-ASCII characters,
 * as well as stripping of any carriage-returns.
 */
template<typename T>
std::pair<std::vector<T>, T> tokenizeRowIO(std::string& line,
                                           int pos = 0,
                                           const std::vector<int>& ignore_col_nums
                                           = empty_int_vec)
{
    table_tokenizer tok = get_row_tokenizer(line);
    std::vector<T> inputs;
    T output;
    int i = 0;
    foreach (const std::string& t, tok) {
        if (boost::find(ignore_col_nums, i) == ignore_col_nums.end()) {
            if (i != pos)
                inputs.push_back(boost::lexical_cast<T>(t));
            else output = boost::lexical_cast<T>(t);
        }
        i++;
    }
    if (pos < 0) {
        output = inputs.back();
        inputs.pop_back();
    }

    return {inputs, output};
}

/**
 * Fill an input table give an istream of DSV file format, where
 * delimiters are ',',' ' or '\t'.
 */
std::istream& istreamITable(std::istream& in, ITable& it,
                            bool has_header, const type_tree& tt);

/**
 * Like above but takes a file_name instead of a istream and
 * automatically infer whether it has header.
 */
void loadITable(const std::string& file_name, ITable& it, const type_tree& tt);

/**
 * Like above but return an ITable and automatically infer the tree
 * tree.
 */
ITable loadITable(const std::string& file_name);

/**
 * Fill an input table and output table given a DSV
 * (delimiter-seperated values) file format, where delimiters are ',',
 * ' ' or '\t'.
 *
 * It is assumed that each row have the same number of columns, if not
 * an assert is raised.
 *
 * pos specifies the position of the output, if -1 it is the last
 * position. The default position is 0, the first column.
 */
std::istream& istreamTable(std::istream& in, ITable& it, OTable& ot,
                           bool has_header, const type_tree& tt, int pos = 0,
                           const std::vector<int>& ignore_col_nums
                           = empty_int_vec);

/**
 * like istreamTable above but take an string (file name) instead of
 * istream. If the file name is not correct then an OC_ASSERT is
 * raised.
 */
void istreamTable(const std::string& file_name,
                  ITable& it, OTable& ot, const type_tree& tt, int pos = 0,
                  const std::vector<int>& ignore_col_nums = empty_int_vec);
/**
 * like above but return an object Table.
 */
Table loadTable(const std::string& file_name, int pos = 0,
                const std::vector<int>& ignore_col_nums = empty_int_vec);

//////////////////
// ostreamTable //
//////////////////

// output the header of a data table in CSV format. target_pos is the
// column index of the target. If -1 then it is the last one.
std::ostream& ostreamTableHeader(std::ostream& out,
                                 const ITable& it, const OTable& ot,
                                 int target_pos = 0);

// output a data table in CSV format. Boolean values are output in
// binary form (0 for false, 1 for true). target_pos is the column
// index of the target. If -1 then it is the last one.
std::ostream& ostreamTable(std::ostream& out,
                           const ITable& it, const OTable& ot,
                           int target_pos = 0);
// like above but take a table instead of an input and output table
std::ostream& ostreamTable(std::ostream& out, const Table& table,
                           int target_pos = 0);

// like above but takes the file name where to write the table
void saveTable(const std::string& file_name,
               const ITable& it, const OTable& ot,
               int target_pos = 0);
// like above but take a table instead of a input and output table
void saveTable(const std::string& file_name, const Table& table,
               int target_pos = 0);

// like ostreamTableHeader but on a compressed table
std::ostream& ostreamCTableHeader(std::ostream& out, const CTable& ct);

// output a compress table in pseudo CSV format
std::ostream& ostreamCTable(std::ostream& out, const CTable& ct);

/**
 * template to subsample input and output tables, after subsampling
 * the table have size min(nsamples, *table.size())
 */
void subsampleTable(ITable& it, OTable& ot, unsigned nsamples);

/**
 * Like above on Table instead of ITable and OTable
 */
void subsampleTable(Table& table, unsigned nsamples);

/**
 * like above but subsample only the input table
 */
void subsampleTable(ITable& it, unsigned nsamples);

/////////////////
// Truth table //
/////////////////

//////////////////////////////
// probably soon deprecated //
//////////////////////////////

// shorthands used by class contin_input_table and contin_output_table
typedef std::vector<bool> bool_vector;
typedef bool_vector::iterator bv_it;
typedef bool_vector::const_iterator bv_cit;
typedef std::vector<bool_vector> bool_matrix;
typedef bool_matrix::iterator bm_it;
typedef bool_matrix::const_iterator bm_cit;

/**
 * complete truth table, it contains only the outputs, the inputs are
 * assumed to be ordered in the conventional way, for instance if
 * there are 2 inputs, the output is ordered as follows:
 *
 * +-----------------------+--+--+
 * |Output                 |$1|$2|
 * +-----------------------+--+--+
 * |complete_truth_table[0]|F |F |
 * +-----------------------+--+--+
 * |complete_truth_table[1]|T |F |
 * +-----------------------+--+--+
 * |complete_truth_table[2]|F |T |
 * +-----------------------+--+--+
 * |complete_truth_table[3]|T |T |
 * +-----------------------+--+--+
 */
class complete_truth_table : public bool_vector
{
public:
    typedef bool_vector super;

    complete_truth_table() {}
    template<typename It>
    complete_truth_table(It from, It to) : super(from, to) {}
    template<typename T>
    complete_truth_table(const tree<T>& tr, arity_t arity)
        : super(pow2(arity)), _arity(arity)
    {
        populate(tr);
    }
    template<typename T>
    complete_truth_table(const tree<T>& tr)
    {
        _arity = arity(tr);
        this->resize(pow2(_arity));
        populate(tr);
    }

    template<typename Func>
    complete_truth_table(const Func& f, arity_t arity)
        : super(pow2(arity)), _arity(arity) {
        iterator it = begin();
        for (int i = 0; it != end(); ++i, ++it) {
            bool_vector v(_arity);
            for (arity_t j = 0;j < _arity;++j)
                v[j] = (i >> j) % 2;
            (*it) = f(v.begin(), v.end());
        }
    }

    /*
      this operator allows to access quickly to the results of a
      complete_truth_table. [from, to) points toward a chain of boolean describing
      the inputs of the function coded into the complete_truth_table and
      the operator returns the results.
    */
    template<typename It>
    bool operator()(It from,It to) {
        const_iterator it = begin();
        for (int i = 1;from != to;++from, i = i << 1)
            if (*from)
                it += i;
        return *it;
    }

    size_type hamming_distance(const complete_truth_table& other) const;

    /**
     * compute the truth table of tr and compare it to self. This
     * method is optimized so that if there are not equal it can be
     * detected before calculating the entire table.
     */
    bool same_complete_truth_table(const combo_tree& tr) const;
protected:
    template<typename T>
    void populate(const tree<T>& tr)
    {
        bmap.resize(_arity);
        iterator it = begin();
        for (int i = 0; it != end(); ++i, ++it) {
            for (int j = 0; j < _arity; ++j)
                bmap[j] = bool_to_vertex((i >> j) % 2);
            *it = eval_binding(bmap, tr) == id::logical_true;
        }
    }
    arity_t _arity;
    mutable vertex_seq bmap;
};

//////////////////
// contin table //
//////////////////

//////////////////////////////
// probably soon deprecated //
//////////////////////////////

//shorthands used by class contin_input_table and contin_output_table
typedef std::vector<contin_t> contin_vector;
typedef contin_vector::iterator cv_it;
typedef contin_vector::const_iterator const_cv_it;
typedef std::vector<contin_vector> contin_matrix;
typedef contin_matrix::iterator cm_it;
typedef contin_matrix::const_iterator const_cm_it;


/**
 * if the DSV data file has a header with labels
 */
std::vector<std::string> readInputLabels(const std::string& file, int pos = 0,
                                         const std::vector<int>& ignore_features
                                         = empty_int_vec);

std::ifstream* open_data_file(const std::string& fileName);

std::ostream& operator<<(std::ostream& out, const ITable& it);

std::ostream& operator<<(std::ostream& out, const OTable& ot);

std::ostream& operator<<(std::ostream& out, const complete_truth_table& tt);

std::ostream& operator<<(std::ostream& out, const CTable& ct);

}} // ~namespaces combo opencog


// TODO see if we can put that under opencog combo
namespace boost
{
inline size_t hash_value(const opencog::combo::complete_truth_table& tt)
{
    return hash_range(tt.begin(), tt.end());
}
} //~namespace boost

#endif // _OPENCOG_TABLE_H
