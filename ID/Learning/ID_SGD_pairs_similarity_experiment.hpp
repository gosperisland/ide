#ifndef _ID_SGD_pairs_similarity_experiment
#define _ID_SGD_pairs_similarity_experiment

#include "Regularization.hpp"
#include "Learning.hpp"
#include "Utils.hpp"

class ID_SGD_pairs_similarity_experiment{
public:

ID_SGD_pairs_similarity_experiment() {}

void dividePrecetage(
    const std::vector<std::vector<double> >& mat,
    const std::vector<short>& tags, 
	std::vector<std::vector<size_t> >& training,				   
	std::vector<short>& training_tags, 
	std::vector<std::vector<size_t> >& testing, 
	std::vector<short>& testing_tags){
	    
    std::srand ( unsigned ( std::time(0) ) );    	
    Learning learn;
    Utils utils;

    std::vector< std::vector<size_t> > pairs = utils.createPairCombinations(mat.size()/2);
    
    assert((mat.size() > 0) && 
    "Number of samples should be greater than zero");
    assert((mat[0].size() > 0) &&
    "Number of features should be greater than zero");
	assert((tags.size() == mat.size()/2) &&
	"tags size should be equal to half of the number of samples");
    assert((tags.size() == pairs.size()) &&
	"tags size should be equal to pairs size");
	
    const size_t NUM_OF_SAMPLES = pairs.size();
    const double TEST_PRECENT = 0.3;
    // size_t temp_test_samples = NUM_OF_SAMPLES * TEST_PRECENT;
    // const size_t NUM_OF_TEST_SAMPLES = temp_test_samples%2 == 0 ? temp_test_samples : temp_test_samples + 1;
    const size_t NUM_OF_TEST_SAMPLES = (size_t)(floor(NUM_OF_SAMPLES * TEST_PRECENT))%2 == 0 ? (size_t)(floor(NUM_OF_SAMPLES * TEST_PRECENT)) : (size_t)(floor(NUM_OF_SAMPLES * TEST_PRECENT)) + 1;
    const size_t NUM_OF_TRAINING_SAMPLES = NUM_OF_SAMPLES - NUM_OF_TEST_SAMPLES;
    const size_t PAIR = 2;

    /*std::cout << "NUM_OF_TEST_SAMPLES: " << NUM_OF_TEST_SAMPLES << 
    ", NUM_OF_TRAINING_SAMPLES: " << NUM_OF_TRAINING_SAMPLES << std::endl;*/
    
    //indecies vector creation
	std::vector<size_t> indecies(pairs.size(), 0);
	iota(indecies.begin(), indecies.end(), 0);  //0,1,2,...pairs.size()-1
	std::random_shuffle ( indecies.begin(), indecies.end() );
    
    testing.resize(NUM_OF_TEST_SAMPLES, std::vector<size_t>(PAIR, 0));
    testing_tags.resize(NUM_OF_TEST_SAMPLES, 0);

    //fill testing vectors
    size_t i;
    for (i = 0; i < NUM_OF_TEST_SAMPLES; i++) {
// 		std::copy ( pairs[indecies[i]].begin(), pairs[indecies[i]].begin() + 1, testing[i].begin() );
        testing[i] = pairs[indecies[i]];
        testing_tags[i] = tags[indecies[i]];
	}
	
    training.resize(NUM_OF_TRAINING_SAMPLES, std::vector<size_t>(PAIR, 0));
    training_tags.resize(NUM_OF_TRAINING_SAMPLES, 0);
    
   // std::cout << "i: " << i << " NUM_OF_SAMPLES: " << NUM_OF_SAMPLES << " " <<std::endl;
   
    //fill training vectors
	for (size_t j = 0 ; i < NUM_OF_SAMPLES; i++) {
// 		std::copy ( pairs[indecies[i]].begin(), pairs[indecies[i]].begin() + 1, training[i].begin() );
        training[j] = pairs[indecies[i]];
        training_tags[j++] = tags[indecies[i]];
	}

    /*
    learn.print2dvector(pairs);     ////////////////////
    learn.printvec(indecies);       ////////////////////
    learn.print2dvector(mat);       ////////////////////
    learn.printvec(tags);           ////////////////////
    learn.print2dvector(testing);   ////////////////////
    learn.printvec(testing_tags);   ////////////////////
    learn.print2dvector(training);  ////////////////////
    learn.printvec(training_tags);  ////////////////////
    size_t a = 0;
    std::cin >> a;
    */
}
	
void divideTraining(
    const std::vector<std::vector<size_t> >& train_mat,
    const std::vector<short>& train_tags, 
	std::vector<std::vector<size_t> >& training,				   
	std::vector<short>& training_tags, 
	std::vector<std::vector<size_t> >& testing, 
	std::vector<short>& testing_tags){
	    
	    std::srand ( unsigned ( std::time(0) ) );    	
	    Learning learn;

	    assert((train_mat.size() > 0) &&
        "Number of samples should be greater than zero");
        assert((train_mat[0].size() > 0) &&
        "Number of features should be greater than zero");
		assert((train_tags.size() == train_mat.size()) &&
		"tags size should be equal to number of samples");
		
	    const size_t NUM_OF_SAMPLES = train_mat.size();
        const double TEST_PRECENT = 0.3;
        const size_t NUM_OF_TEST_SAMPLES = (size_t)(floor(NUM_OF_SAMPLES * TEST_PRECENT))%2 == 0 ? (size_t)(floor(NUM_OF_SAMPLES * TEST_PRECENT)) : (size_t)(floor(NUM_OF_SAMPLES * TEST_PRECENT)) + 1;
        const size_t NUM_OF_TRAINING_SAMPLES = NUM_OF_SAMPLES - NUM_OF_TEST_SAMPLES;
        const size_t PAIR = 2;

        /*std::cout << "NUM_OF_TEST_SAMPLES: " << NUM_OF_TEST_SAMPLES << 
        ", NUM_OF_TRAINING_SAMPLES: " << NUM_OF_TRAINING_SAMPLES << std::endl;*/
        
	    //indecies vector creation
    	std::vector<size_t> indecies(train_mat.size(), 0);
    	iota(indecies.begin(), indecies.end(), 0);  //0,1,2,...pairs.size()-1
    	std::random_shuffle ( indecies.begin(), indecies.end() );
        
        testing.resize(NUM_OF_TEST_SAMPLES, std::vector<size_t>(PAIR, 0));
        testing_tags.resize(NUM_OF_TEST_SAMPLES, 0);

	    //fill testing vectors
	    size_t i;
        for (i = 0; i < NUM_OF_TEST_SAMPLES; i++) {
            testing[i] = train_mat[indecies[i]];
            testing_tags[i] = train_tags[indecies[i]];
    	}
    	
        training.resize(NUM_OF_TRAINING_SAMPLES, std::vector<size_t>(PAIR, 0));
        training_tags.resize(NUM_OF_TRAINING_SAMPLES, 0);
	    
	   // std::cout << "i: " << i << " NUM_OF_SAMPLES: " << NUM_OF_SAMPLES << " " <<std::endl;
	   
	    //fill training vectors
    	for (size_t j = 0 ; i < NUM_OF_SAMPLES; i++) {
            training[j] = train_mat[indecies[i]];
            training_tags[j++] = train_tags[indecies[i]];
    	}

        /*
        learn.printvec(indecies);       ////////////////////
        learn.print2dvector(train_mat); ////////////////////
        learn.printvec(train_tags);     ////////////////////
        learn.print2dvector(testing);   ////////////////////
        learn.printvec(testing_tags);   ////////////////////
        learn.print2dvector(training);  ////////////////////
        learn.printvec(training_tags);  ////////////////////
        size_t a = 0;
        std::cin >> a;
        */
	}
	
double find_best_C(
    const std::vector<std::vector<double> >& examples, 
    const std::vector<std::vector<size_t> >& train_mat, 
    const std::vector<short> train_tags, 
    const std::vector<std::vector<double> > _discrete_points, 
    const std::vector<std::vector<size_t> >& indices_of_groups, 
    IDpair& id_pair, 
    const bool if_equal_dist_zero, 
    const bool if_non_equal_dist_non_zero, 
    const double NON_EQUAL_EPSILON, 
    const bool symmetry, 
    const std::vector<double>& Wreg, 
    std::vector<double>& W, 
    const std::vector<double>& C, 
    double& thold){
    
    Learning learn;
    ID_SGD_pairs_similarity_experiment id_exp;

    std::vector<std::vector<size_t> > training;				   
	std::vector<short> training_tags;
	std::vector<std::vector<size_t> > testing;
	std::vector<short> testing_tags;
	

    const size_t ITERATION = 5;
    size_t numOfErrors = 0;
    std::vector<double> avgErr(C.size(), 0);
    
    std::cout << "before W: " << std::endl;
    learn.printvec(W);
        
    for (size_t i = 0; i < C.size(); i++) {
        std::vector<size_t> error_vec(ITERATION, 0);
	    for (size_t j = 0; j < ITERATION; j++) {
	        
    	    id_exp.divideTraining(train_mat, train_tags, training, training_tags, testing, testing_tags);
            //W.resize(Wreg.size());
            std::fill (W.begin(), W.end(), 0);
            
            learn(examples, training, training_tags, _discrete_points, indices_of_groups,
                       id_pair, if_equal_dist_zero, if_non_equal_dist_non_zero, 
                       NON_EQUAL_EPSILON, symmetry, Wreg, C[i], W, thold);
            
            for (size_t j = 0; j < testing.size(); j++) {
                std::vector<Pair> vol = id_pair( examples[ testing[j][0] ], examples[ testing[j][1] ] );
                short s = Learning::classify(W, vol, thold);
        
                if(s != testing_tags[j]) numOfErrors++;
            }
            
            error_vec[j] = numOfErrors;
            
            std::cout << "C[" << i << "] = " << C[i] << std::endl; 
            std::cout << "amount of errors: "<< numOfErrors << " out of " << testing_tags.size() <<  " tests" << std::endl;
            std::cout << "Success rate: " << (1 - (double)numOfErrors/testing_tags.size()) << std::endl; 
            std::cout << "thold: " << thold << std::endl;
            numOfErrors = 0;
            thold = 1;
            
        }
        
        std::cout << "W: " << std::endl;
        learn.printvec(W);
        avgErr[i] = std::accumulate(error_vec.begin(), error_vec.end(), 0); //sum error_vec values
        avgErr[i] /= ITERATION;
        std::cout << "avgErr[" << i << "]: " << avgErr[i] << "\n\n" << std::endl;
    }
    
    size_t min_index = std::min_element(avgErr.begin(), avgErr.end()) - avgErr.begin(); //find min value index in avgErr 
	double best_C = C[min_index];
	/*
	std::cout << "min index: " << min_index << " C[min_index]: " << C[min_index] << " best_C: " << best_C <<
	", thold:" << thold << "\n" << std::endl;
	*/
	
	
	learn   (examples, training, training_tags, _discrete_points, indices_of_groups,
            id_pair, if_equal_dist_zero, if_non_equal_dist_non_zero, 
            NON_EQUAL_EPSILON, symmetry, Wreg, best_C, W, thold);
               
	learn.printvec(W);
	
	return best_C;
}

};
#endif