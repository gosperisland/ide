#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cassert>
#include <exception>
#include <time.h>
#include "armadillo"
#include "Learning.hpp"
#include "/home/ubuntu/workspace/New/GridsOfGroups.hpp"
using namespace arma;

void testClassificationPairs() {
    const size_t numOfSamples = 2000;
    std::vector<double> gridForX1 = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<double> gridForX2 = {0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
   
    std::vector<std::vector<double> > descrete_points = {gridForX1, gridForX2};
    std::vector<std::vector<size_t> > indices_of_groups = { {0,1}, {1,0} };
    std::vector<std::vector<double> > examples;
    
    const bool symmetry = true;
    const bool if_equal_dist_zero = true;
	const bool if_non_equal_dist_non_zero = true;
	const double NON_EQUAL_EPSILON = std::numeric_limits<double>::epsilon();
	
    arma_rng::set_seed_random();
    vec A = randi<vec>(numOfSamples/2, distr_param(0, 100));
    vec B = randi<vec>(numOfSamples/2, distr_param(0, 200));
    
    for (size_t i = 0; i < A.size(); ++i) {
        std::vector<double> p1 = {A(i), B(i)};
        examples.push_back(p1);
    }
    
    A = randi<vec>(numOfSamples/2, distr_param(0, 100));
    B = randi<vec>(numOfSamples/2, distr_param(0, 200));
    
    for (size_t i = 0; i < B.size(); ++i) {
        std::vector<double> p1 = {A(i), B(i)};
        examples.push_back(p1);
    }

    std::vector<std::vector <size_t> > pairs_of_indices(numOfSamples/2);

    for (size_t i = 0; i < numOfSamples/2; i++) {
        std::vector<size_t> p1 = {i, i + numOfSamples/2};
        pairs_of_indices[i] = p1;
    }

    Regularization regularization;
    std::vector<short> tags(numOfSamples/2,0);
    size_t counter = 0;
    for (size_t i = 0; i < pairs_of_indices.size(); i++) {
        double dist = regularization.L1DistanceScalar( examples[pairs_of_indices[i][0]][0], examples[pairs_of_indices[i][1]][0] );

        tags[i] = dist < 50 ? -1 : 1;
        if(tags[i] < 0) counter++;
        
    }
    std::cout<<"counter: "<< counter << std::endl;
    
    std::vector<double> C = regularization.c_vec_intialization();
    double thold = 1;
    size_t numOfErrors = 0;
    Grid grid(descrete_points);
    IDpair id_pair(grid);
    grid = id_pair.get_grid(); //the Grid of id_pair is passed by value
    std::vector<double> Wreg = regularization.construct_Wreg(grid);
    std::vector<double> W(Wreg.size(), 0);
    Learning learning;
    
    for (auto c : C) {
        learning.SGD(
            examples, pairs_of_indices, tags, descrete_points, indices_of_groups, 
            id_pair, if_equal_dist_zero, if_non_equal_dist_non_zero, 
            NON_EQUAL_EPSILON, symmetry, Wreg, c, W, thold);
        
        for (size_t i = 0; i < pairs_of_indices.size(); i++) {
            std::vector<Pair> vol = id_pair( examples[ pairs_of_indices[i][0] ], examples[ pairs_of_indices[i][1] ] );
            short s = learning.classification(W, vol, thold);
            if(s != tags[i]) numOfErrors++;
        }
        
        std::cout<< "c : " << c << std::endl; 
        std::cout << "num of errors: " << numOfErrors << std::endl;
        std::cout << "errors precent: " << (double)numOfErrors/tags.size() << std::endl; 
        std::cout << "thold: " << thold << "\n" << std::endl;
        numOfErrors = 0;
        thold = 1;
    }
}

void sanityTest1(){
    std::vector<std::vector<double> > dis_points = {{0, 2, 3, 4, 5, 6}, {0, 2, 3, 4, 5, 6}};
    std::vector<std::vector<double> > for_grid_pair(dis_points);
    Grid grid(for_grid_pair);
    IDpair id_pair(grid);
    Grid grid_pair = id_pair.get_grid();
    std::vector<double> vec1 = {1.5, 3.5};
    std::vector<double> vec2 = {4, 2};
    std::vector<Pair> result = id_pair(vec1, vec2);
    
    Regularization regularization;
    std::vector<double> w = regularization.construct_Wreg(grid_pair);
    double dotProd = 0;
    
    for (size_t i = 0; i < result.size() ; i++)
        dotProd += result[i]._weight * w[ result[i]._index ];

    std::cout << "dot: " << dotProd << std::endl;
}

void sanityTest2(){
    // implement sanity tests (metric but not eaclidean), page 26 on ID document.
    std::vector<std::vector<double> > grid = {{0, 1, 2}, {0, 1, 2}};
    std::vector<std::vector<double> > gridpair(grid);
    gridpair.insert(gridpair.end(), grid.begin(), grid.end());
    
    std::vector<double> vec1 = {0, 0};
    std::vector<double> vec2 = {2, 0};
    std::vector<double> vec3 = {0, 2};
    std::vector<double> vec4 = {1, 1};
    
    std::vector<double> tags = {2, 1, 1, 2, 2, 1};
    Regularization regularization;
    
    tags.push_back(regularization.L1Distance(vec1, vec2));
    tags.push_back(regularization.L1Distance(vec1, vec3));
    tags.push_back(regularization.L1Distance(vec1, vec4));
    tags.push_back(regularization.L1Distance(vec2, vec3));
    tags.push_back(regularization.L1Distance(vec2, vec4));
    tags.push_back(regularization.L1Distance(vec3, vec4));
    
    for (auto i : tags) {
        std::cout << i << " , ";
    }std::cout<<std::endl;
}


// compile::  g++ -Werror -Wall -Wextra -std=c++11 euclideanDis.cpp -o T
int main(){
    
    testClassificationPairs();
    
    // sanityTest1();

    // sanityTest2();
    
    return 0;
}