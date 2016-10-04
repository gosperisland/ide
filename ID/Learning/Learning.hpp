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
    static void printvec(const std::vector<T>& v){
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
		
        srand(time(NULL));  
        assert((examples.size() > 0) &&
        "Number of samples should be greater than zero");
        assert((examples[0].size() > 0) &&
        "Number of features should be greater than zero");
		assert((labels.size() == pairs_of_indices.size()) &&
		"labels size should be equal to pairs_of_indices size");
        assert((Wreg.size() == W.size()) &&
        "Wreg size should be equal to W size");

        std::cout.setstate(std::ios_base::failbit);                                             //// discard any output
        std::cout << discrete_points_for_all_dims.size() << indices_of_groups.size() << std::endl;
        std::cout.clear();                                                                      /////// get Output
        
		/* TODO
		//GridsOfGroups(const std::vector<std::vector<double> >& discrete_points_for_all_dims, const std::vector<std::vector<size_t> >& indices_of_groups) 
		//IDgroupsPair(const GridsOfGroups& grids_of_groups, const std::vector<std::vector<size_t> >& indices_of_groups) */
		
		_if_equal_dist_zero = if_equal_dist_zero;
		_if_non_equal_dist_non_zero = if_non_equal_dist_non_zero;
		_NON_EQUAL_EPSILON = NON_EQUAL_EPSILON;
		_symmetry = symmetry;
        
        const size_t NUMBER_OF_ITERATIONS = EPOCH_TIMES * (examples.size());
        for (size_t j = 0; j < NUMBER_OF_ITERATIONS; ++j) {
            std::vector<size_t> random_indexes(pairs_of_indices.size()) ;
            std::iota (std::begin(random_indexes), std::end(random_indexes), 0);
            std::random_shuffle ( random_indexes.begin(), random_indexes.end() );
            // size_t random_index = 0 + rand() % pairs_of_indices.size();

            for (size_t i = 0; i < pairs_of_indices.size(); i++) {
                size_t random_index = random_indexes.back();
                random_indexes.pop_back();
                //size_t random_index = std::rand() % pairs_of_indices.size();
                std::vector<double> first_exam = examples[ pairs_of_indices[random_index][0] ];
                std::vector<double> second_exam = examples[ pairs_of_indices[random_index][1] ];
                SGD_similar(id_pair, W, Wreg, first_exam, second_exam, labels[random_index], C, i + 1, thold);
            }
            /*size_t a;
            std::cout << "**************" << std::endl;
            std::cin >> a;*/
        }
        /*std::cout << "W:\n";
        printvec(W);
        std::cout << "Wreg:\n";
        printvec(Wreg);*/
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
        /*std::cout << "first_exam: ";
        printvec(first_exam);
        std::cout << "second_exam: ";
        printvec(second_exam);*/
        _if_equal_dist_zero = first_exam==second_exam;
        _if_non_equal_dist_non_zero = !_if_equal_dist_zero;
        // std::cout << "first==second: " << _if_equal_dist_zero << " , tag: " << tag << "\n\n" << std::endl;
        assert((_if_non_equal_dist_non_zero || tag == -1) 
        && "The samples are exactly the same tag should be \"-1\" for Similar objects, check_input()");
        return _if_equal_dist_zero;
    }
    
    static void loss(
        IDpair& id_pair,
        const std::vector<double>& first_exam,  
        const std::vector<double>& second_exam, 
        std::vector<double>& W,
        double& thold,
        const short tag, 
        const double C, 
        const size_t step){
            
        const std::vector<Pair>& embedded_vec = id_pair(first_exam, second_exam);
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
        
        if(check_input(first_exam, second_exam, tag)){
            // bool a;
            
            const std::vector<Pair>& embedded_vec = id_pair(first_exam, second_exam);
            for (auto simplex_point : embedded_vec) {
            //   std::cout << "index: " << simplex_point._index << " , weight: " << simplex_point._weight << " , W[index]:" << W[simplex_point._index] << std::endl;
              W[simplex_point._index] = 0;
            }

            /*std::cout << "W:\n";
            printvec(W);
            std::cout << "Wreg:\n";
            printvec(Wreg);
            std::cin >> a;*/
            return;  
        } 

        std::vector<double> W_old(W);   
        loss(id_pair, first_exam, second_exam, W, thold, tag, C, step);
/*        const std::vector<Pair>& embedded_vec = id_pair(first_exam, second_exam);
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
        }*/
        for (size_t i = 0; i < W.size(); i++){
            double grad_mult_stepsize = (1.0/(double)step) * ((W_old[i] - Wreg[i]));
            update_W(W, i, grad_mult_stepsize);
        }
    }
};

bool Learning::_if_equal_dist_zero(true);
bool Learning::_if_non_equal_dist_non_zero(true);
bool Learning::_symmetry(true);
double Learning::_NON_EQUAL_EPSILON(std::numeric_limits<double>::epsilon());

#endif
