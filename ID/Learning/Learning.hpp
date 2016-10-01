#ifndef _LEARNING_ID
#define _LEARNING_ID

#include "/home/ubuntu/workspace/ID/IDpair.hpp"

#define ALPHA 1.0
#define EPOCH_TIMES 3

// compile: g++ -g -Wall -Werror -Wextra -Wfatal-errors -std=c++11 euclideanDis.cpp -o T
class Learning{

public:

    Learning() {}
    
    template <typename T>
    void printvec(const std::vector<T>& v){
        for (auto i : v) {
            std::cout << i << " ";
        }std::cout << "\n" << std::endl;
    }
    
    template <typename T>
    void print2dvector(const std::vector<std::vector<T> >& v) {
        for (auto i : v) {
            for (auto j : i) {
                std::cout << j << " ";
            }std::cout << std::endl;
        }std::cout << std::endl;
    }
    
    static short classify(const std::vector<double>& W, const std::vector<Pair>& non_zero, double thold){
        double dotProd = 0;
        short ans = 0;

        for (size_t i = 0; i < non_zero.size() ; i++)
            dotProd += non_zero[i]._weight * W[ non_zero[i]._index ];
        
        // std::cout << "dotprod: " << dotProd << " , thold: " << thold << std::endl;
        if( (dotProd - thold) < 0) ans = 1;
        else ans = -1;

        return ans;
    }


void operator()(
		const std::vector<std::vector <double> >& examples,
		const std::vector<std::vector <size_t> >& pairs_of_indices,
		const std::vector<short>& labels,

		// ID discritization params
		const std::vector<std::vector<double> >& discrete_points_for_all_dims,
		const std::vector<std::vector<size_t> >& indices_of_groups,
		IDpair& id_pair,

		// ID metric properties
		const bool if_equal_dist_zero,
		const bool if_non_equal_dist_non_zero,
		const double NON_EQUAL_EPSILON,
		const bool symmetry,

		// Regulrization params
		const std::vector<double>& Wreg,
		const double C,

		// Output params
		std::vector<double>& W,
		double& thold){
		
        assert((examples.size() > 0) &&
        "Number of samples should be greater than zero");
        assert((examples[0].size() > 0) &&
        "Number of features should be greater than zero");
		assert((labels.size() == pairs_of_indices.size()) &&
		"labels size should be equal to pairs_of_indices size");
        assert((Wreg.size() == W.size()) &&
        "Wreg size should be equal to W size");

		/* TODO
		//GridsOfGroups(const std::vector<std::vector<double> >& discrete_points_for_all_dims, const std::vector<std::vector<size_t> >& indices_of_groups) 
		//IDgroupsPair(const GridsOfGroups& grids_of_groups, const std::vector<std::vector<size_t> >& indices_of_groups) */
		
		_if_equal_dist_zero = if_equal_dist_zero;
		_if_non_equal_dist_non_zero = if_non_equal_dist_non_zero;
		_NON_EQUAL_EPSILON = NON_EQUAL_EPSILON;
		_symmetry = symmetry;
        
        std::cout << "before operator()" << std::endl;
        const size_t NUMBER_OF_ITERATIONS = EPOCH_TIMES * (examples.size());
        for (size_t j = 0; j < NUMBER_OF_ITERATIONS; ++j) {
            std::vector<size_t> random_indexes(pairs_of_indices.size()) ;
            std::iota (std::begin(random_indexes), std::end(random_indexes), 0);
            std::random_shuffle ( random_indexes.begin(), random_indexes.end() );

            for (size_t i = 0; i < pairs_of_indices.size(); i++) {
                size_t random_index = random_indexes.back();
                random_indexes.pop_back();
                std::vector<double> first_exam = examples[ pairs_of_indices[random_index][0] ];
                std::vector<double> second_exam = examples[ pairs_of_indices[random_index][1] ];
                SGD_similar(id_pair, W, Wreg, first_exam, second_exam, labels[random_index], C, i + 1, thold);
            }
        }
        std::cout << "after operator()" << std::endl;
    }
    

private:
    static bool _if_equal_dist_zero;
    static bool _if_non_equal_dist_non_zero;
    static bool _symmetry;
    static double _NON_EQUAL_EPSILON;
    
    static void update_W(std::vector<double>& W, const size_t index, const double value){
        W[index] -= value;
        W[index] = W[index] < _NON_EQUAL_EPSILON ? _NON_EQUAL_EPSILON : W[index];
    }
    
    /**
     *  TODO:: check if these fields: _if_equal_dist_zero, _if_non_equal_dist_non_zero need to be changed here
     */
    static bool check_input( const std::vector<double>& first_exam, const std::vector<double>& second_exam, const short tag){
            assert((first_exam != second_exam || tag == 1) && "check_input()");
            return first_exam == second_exam;
    }
        
    static void SGD_similar(
        IDpair& id_pair,
        std::vector<double>& W,
        const std::vector<double>& Wreg, 
        const std::vector<double>& first_exam,  
        const std::vector<double>& second_exam, 
        const short tag, 
        const double C, 
        const size_t step,
        double& thold){
        
        if(check_input(first_exam, second_exam, tag)) return;
        
        const std::vector<Pair>& embedded_vec = id_pair(first_exam, second_exam);
        std::vector<double> W_old(W);   
        double dotProd = 0;

        //ID(X_pi_1, X_pi_2) * W
        for (size_t i = 0; i < embedded_vec.size(); i++)
            dotProd += embedded_vec[i]._weight * W[ embedded_vec[i]._index ];

        // 1 - { (ID(X_pi_1, X_pi_2) * W) - threshold } * y_i
        if(  ( 1 - ( (dotProd - thold) * tag) ) > 0 ) {
            for (auto& simplex_point : embedded_vec) {
                double grad_mult_stepsize = -(1.0/(double)step) *  (C * tag * simplex_point._weight);
                update_W(W, simplex_point._index, grad_mult_stepsize);
            }
            thold -= ( (1.0 / (double)step) * (C * tag) );
            thold = thold < 1 ? 1 : thold;
        }

        for (size_t i = 0; i < W.size(); i++){
            double grad_mult_stepsize = (1.0/(double)step) * ((W_old[i] - Wreg[i]));
            update_W(W, i, grad_mult_stepsize);
        }
        std::cout << "after SGD_similar()" << std::endl;

    }
};

bool Learning::_if_equal_dist_zero(true);
bool Learning::_if_non_equal_dist_non_zero(true);
bool Learning::_symmetry(true);
double Learning::_NON_EQUAL_EPSILON(std::numeric_limits<double>::epsilon());

#endif
