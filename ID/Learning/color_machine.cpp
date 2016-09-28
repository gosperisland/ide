#include "ID_SGD_pairs_similarity_experiment.hpp"

double code_to_time(){
    Utils utils;
    Regularization reg;
    Learning learn;

    //read original data to matrix
    arma::mat data = utils.load_file_to_matrix("/home/ubuntu/workspace/New/Learning_Tests_FIX/colors_v1.csv");
    
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
    
  /*  for(uword row=0; row < data_for_disc.n_rows; ++row){
        for(uword col=0; col < data_for_disc.n_cols; ++col){
            cout << data_for_disc(row,col) << ' '; 
        }std::cout << std::endl;
    }std::cout << "\n\n" << std::endl;
  */  
    //find discrit points
    arma::mat dis_points = utils.find_discrit_points(data_for_disc,5);
    
/*    for(uword row=0; row < dis_points.n_rows; ++row){
        for(uword col=0; col < dis_points.n_cols; ++col){
            cout << dis_points(row,col) << ' '; 
        }std::cout << std::endl;
    }*/
    
    //generate pairs
    std::vector< std::vector<size_t> > pairs = utils.create_combination_2(data.n_rows);
    learn.print2dvector(pairs);
    
    size_t counter_minus = 0;
    size_t counter_plus = 0;

    std::vector<short> tags(pairs.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        short label = y_tag[i] != 1 ? -1 : 1;
        tags[i] = label;

        if(label == -1){
            counter_minus++;
        }else{
            counter_plus++;
        }
    }
    
    learn.printvec(tags);


    std::cout << "unSimilar ratio: " << (double)counter_plus/(double)pairs.size() << std::endl;
    std::cout << "Similar ratio: " << (double)counter_minus/(double)pairs.size() << std::endl;
    
    double thold = 1;
    std::vector<double> C = reg.c_vec_intialization();
    
    //get matrix sorted by rows [3][5]
    std::vector< std::vector<double> > dis_points2 = utils.mat_to_std_vec_2(dis_points);
    
    std::vector< std::vector<double> > examples = utils.mat_to_std_vec(data_for_disc);

    //learn.print2dvector(examples);      /////////////////////////
    //learn.printvec(tags);               /////////////////////////
    
    
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

    
    for (size_t i = 0; i < C.size(); i++) {
        std::vector<double> W(Wreg.size(), 0);
        
        learn(examples, pairs, tags, dis_points2,indices_of_groups,
                   id_pair, if_equal_dist_zero, if_non_equal_dist_non_zero, 
                   NON_EQUAL_EPSILON, symmetry, Wreg, C[i], W, thold);
    
        for (size_t j = 0; j < pairs.size(); j++) {
            std::vector<Pair> vol = id_pair( examples[ pairs[j][0] ], examples[ pairs[j][1] ] );
            short s = learn.classify(W, vol, thold);
    
            if(s != tags[j]) numOfErrors++;
        }
        
        std::cout << "C[" << i << "] = " << C[i] << std::endl; 
        std::cout << "num of errors: " << numOfErrors << std::endl;
        std::cout << "errors precent: " << (double)numOfErrors/tags.size() << std::endl; 
        std::cout << "thold: " << thold << "\n\n" << std::endl;
        numOfErrors = 0;
        thold = 1;
        
        // learn.printvec(W);          ///////////////////////////
    }
    
    return  0;
}

int main() {
    clock_t begin = clock();

  code_to_time();

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  
  std::cout << "Time: " << elapsed_secs << std::endl;
  
  return 0;
}