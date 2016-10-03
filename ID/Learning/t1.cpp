#include "ID_SGD_pairs_similarity_experiment.hpp"

//alias runcol='g++ -g -std=c++11 -Wall color_machine.cpp -o T ; sleep 1 ; ./T'
//alias runt1='g++ -g -std=c++11 -Wall t1.cpp -o T1 ; sleep 1 ; ./T1'
// compile: runt1
// run: ./T1

int main(){
    
    clock_t begin = clock();
    // std::cout.setstate(std::ios_base::failbit);     /////////// discard any output

    Utils u;
    Learning learn;
    Regularization reg;
    ID_SGD_pairs_similarity_experiment id_exp;
    
    u.readCSV("/home/ubuntu/workspace/ID/Learning/colors_v1.csv");
    learn.print2dvector(u._discrete_points_for_all_dims);
    /*arma::mat data = u.loadFileToMatrix("/home/ubuntu/workspace/ID/Learning/colors_v1.csv");
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
    
    arma::mat dis_points = u.findDiscretePoints(data_for_disc,5);
    const std::vector<std::vector<double> > _discrete_points_for_all_dims = u.matToVecByColumns(dis_points);
    const std::vector<std::vector<double> > examples = u.matToVecByRows(data_for_disc);*/
    
	/*std::cout << " **** after division ****\n" << std::endl;
    for(arma::uword row=0; row < dis_points.n_rows; ++row){
        for(arma::uword col=0; col < dis_points.n_cols; ++col){
            std::cout << dis_points(row,col) << ' '; 
        }std::cout << std::endl;
    }std::cout << std::endl;
    
    learn.print2dvector(_discrete_points_for_all_dims);  ////////////////////
*/

    double thold = 1;
	size_t numOfErrors = 0;
    const bool symmetry = true;
    const bool if_equal_dist_zero = true;
	const bool if_non_equal_dist_non_zero = true;
	const double NON_EQUAL_EPSILON = std::numeric_limits<double>::epsilon();
    const std::vector<std::vector<size_t> > indices_of_groups = { {0,1}, {1,0} };
    Grid grid_pair(u._discrete_points_for_all_dims);
    IDpair id_pair(grid_pair);
    grid_pair = id_pair.get_grid();
    const std::vector<double> Wreg = reg.construct_Wreg(grid_pair);
    const std::vector<double> C = reg.c_vec_intialization();
    std::vector<std::vector<size_t> > training;				   
	std::vector<short> training_tags;
	std::vector<std::vector<size_t> > testing;
	std::vector<short> testing_tags;
    std::vector<double> W(Wreg.size(), 0);
    
    std::cout << "W.size: " << W.size() << std::endl;
    
    id_exp.dividePrecetage(u._examples, u._tags, training, training_tags, testing, testing_tags);
    
    double c = id_exp.find_best_C (u._examples, training, training_tags, u._discrete_points_for_all_dims, indices_of_groups, id_pair, 
                if_equal_dist_zero, if_non_equal_dist_non_zero, NON_EQUAL_EPSILON, symmetry, Wreg, W, C, thold );
                
    for (size_t j = 0; j < testing.size(); j++) {
        std::vector<Pair> vol = id_pair( u._examples[ testing[j][0] ], u._examples[ testing[j][1] ] );
        short s = Learning::classify(W, vol, thold);

        if(s != testing_tags[j]) numOfErrors++;
    }

    // std::cout.clear();          /////// get Output

    
    std::cout << "Best C: " << c << std::endl; 
    std::cout << "amount of errors: " << numOfErrors << " out of " << testing_tags.size() <<  " tests" << std::endl;
    std::cout << "Success rate: " << (1 - (double)numOfErrors/testing_tags.size()) * 100 << "%" << std::endl; 
    std::cout << "thold: " << thold << "\n\n" << std::endl;
    
    // learn.printvec(W);          ///////////////////////////
    
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    
    std::cout << "Time: " << elapsed_secs << std::endl;
    
    return 0;
}