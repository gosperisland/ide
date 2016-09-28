#include "ID_SGD_pairs_similarity_experiment.hpp"

//alias runcol='g++ -g -std=c++11 -Wall color_machine.cpp -o T ; sleep 1 ; ./T'
//alias runt1='g++ -g -std=c++11 -Wall t1.cpp -o T1 ; sleep 1 ; ./T1'
// compile: runt1
// run: ./T1

int main(){
    
    Utils u;
    Learning learn;
    Regularization reg;
    ID_SGD_pairs_similarity_experiment id_devide;
    
    arma::mat data = u.load_file_to_matrix("/home/ubuntu/workspace/ID/Learning/colors_v1.csv");
    arma::mat x1(data.n_rows, 3);
    arma::mat x2(data.n_rows, 3);
    
    for (size_t i = 0; i < 3; ++i) 
        x1.col(i) = data.col(i);
    for (size_t i = 5; i >= 3; --i) 
        x2.col(i-3) = data.col(i);

    arma::mat y_tag(data.n_rows, 1);
    y_tag.col(0) = data.col(6);
    
    arma::mat data_for_disc(2*data.n_rows, 3);
    
    for (size_t j = 0; j < x1.n_rows; ++j) 
        data_for_disc.row(j) = x1.row(j);
    for (size_t j = x2.n_rows; j < 2*(x2.n_rows); ++j) 
        data_for_disc.row(j) = x2.row(j-x2.n_rows);
        
    std::vector<short> tags(data.n_rows, 0);
    for (size_t i = 0; i < tags.size(); ++i) 
        tags[i] = y_tag[i] != 1 ? -1 : 1;
    
    arma::mat dis_points = u.find_discrit_points(data_for_disc,5);
    std::vector<std::vector<double> > dis_points2 = u.mat_to_std_vec_2(dis_points);
    std::vector<std::vector<double> > examples = u.mat_to_std_vec(data_for_disc);

	std::vector<std::vector<size_t> > training;				   
	std::vector<short> training_tags;
	std::vector<std::vector<size_t> > testing;
	std::vector<short> testing_tags;
	
	id_devide.dividePrecetage(examples, tags, training, training_tags, testing, testing_tags);
	
	std::cout << " **** after division ****\n" << std::endl;
    for(arma::uword row=0; row < dis_points.n_rows; ++row){
        for(arma::uword col=0; col < dis_points.n_cols; ++col){
            std::cout << dis_points(row,col) << ' '; 
        }std::cout << std::endl;
    }std::cout << std::endl;
    
    learn.print2dvector(dis_points2);  ////////////////////

    double thold = 1;
	size_t numOfErrors = 0;
    const bool symmetry = true;
    const bool if_equal_dist_zero = true;
	const bool if_non_equal_dist_non_zero = true;
	const double NON_EQUAL_EPSILON = std::numeric_limits<double>::epsilon();
    std::vector<std::vector<size_t> > indices_of_groups = { {0,1}, {1,0} };
    Grid grid_pair(dis_points2);
    IDpair id_pair(grid_pair);
    grid_pair = id_pair.get_grid();
    std::vector<double> Wreg = reg.construct_Wreg(grid_pair);
    std::vector<double> C = reg.c_vec_intialization();

    
    for (size_t i = 0; i < C.size(); i++) {
        std::vector<double> W(Wreg.size(), 0);
        
        learn(examples, training, training_tags, dis_points2, indices_of_groups,
                   id_pair, if_equal_dist_zero, if_non_equal_dist_non_zero, 
                   NON_EQUAL_EPSILON, symmetry, Wreg, C[i], W, thold);
                   
        for (size_t j = 0; j < testing.size(); j++) {
            std::vector<Pair> vol = id_pair( examples[ testing[j][0] ], examples[ testing[j][1] ] );
            short s = Learning::classify(W, vol, thold);
    
            if(s != testing_tags[j]) numOfErrors++;
        }
        
        std::cout << "C[" << i << "] = " << C[i] << std::endl; 
        std::cout << "num of errors: " << numOfErrors << std::endl;
        std::cout << "errors precent: " << (double)numOfErrors/testing_tags.size() << std::endl; 
        std::cout << "thold: " << thold << "\n\n" << std::endl;
        numOfErrors = 0;
        thold = 1;
        return 0;
        // learn.printvec(W);          ///////////////////////////
    }
    
    return 0;
}