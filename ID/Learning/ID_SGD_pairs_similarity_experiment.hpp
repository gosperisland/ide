#ifndef _ID_SGD_pairs_similarity_experiment
#define _ID_SGD_pairs_similarity_experiment

#include "Regularization.hpp"
#include "Learning.hpp"
#include "Utils.hpp"

class ID_SGD_pairs_similarity_experiment{
public:

ID_SGD_pairs_similarity_experiment() {}

template <typename T>
void convertLables(const std::vector<T> &labels){
    std::vector<T> tags(labels);
	for (size_t i = 0; i < tags.size(); i++)
	    tags[i] = tags[i] != 1 ? -1 : 1;
}

void dividePrecetage(
    const std::vector<std::vector<double> > &mat,
    const std::vector<short> &tags, 
	std::vector<std::vector<size_t> > &training,				   
	std::vector<short> &training_tags, 
	std::vector<std::vector<size_t> > &testing, 
	std::vector<short> &testing_tags){
	    
	    std::srand ( unsigned ( std::time(0) ) );    	
	    Learning learn;
	    Utils utils;

        std::vector< std::vector<size_t> > pairs = utils.create_combination_2(mat.size()/2);
        
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

        learn.print2dvector(pairs);     ////////////////////
        learn.printvec(indecies);       ////////////////////
        learn.print2dvector(mat);       ////////////////////
        learn.printvec(tags);           ////////////////////
        learn.print2dvector(testing);   ////////////////////
        learn.printvec(testing_tags);   ////////////////////
        learn.print2dvector(training);  ////////////////////
        learn.printvec(training_tags);  ////////////////////
        
	}
};
#endif