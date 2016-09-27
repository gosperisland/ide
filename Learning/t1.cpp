#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cassert>
#include <exception>
#include <math.h>
#include <time.h>
#include "Utils.hpp"
#include "Learning.hpp"
#include "ID_SGD_pairs_similarity_experiment.hpp"

// using namespace std;
// #include <algorithm>    // std::replace_if

template <typename Iterator>
inline bool next_combination(const Iterator first, Iterator k, const Iterator last)
{
   if ((first == last) || (first == k) || (last == k))
      return false;
   Iterator itr1 = first;
   Iterator itr2 = last;
   ++itr1;
   if (last == itr1)
      return false;
   itr1 = last;
   --itr1;
   itr1 = k;
   --itr2;
   while (first != itr1)
   {
      if (*--itr1 < *itr2)
      {
         Iterator j = k;
         while (!(*itr1 < *j)) ++j;
         std::iter_swap(itr1,j);
         ++itr1;
         ++j;
         itr2 = k;
         std::rotate(itr1,j,last);
         while (last != j)
         {
            ++j;
            ++itr2;
         }
         std::rotate(k,itr2,last);
         return true;
      }
   }
   std::rotate(first,k,last);
   return false;
}


bool IsOdd (short i) { return (i!=-1); }

// compile: g++ -g -std=c++11 -Wall t1.cpp -o T1
int main(){
    
    Utils u;
    Learning learn;
    Regularization reg;
    ID_SGD_pairs_similarity_experiment id_devide;
    
    mat data = u.load_file_to_matrix("/home/ubuntu/workspace/New/Learning_Tests_FIX/colors_v1.csv");
    mat x1(data.n_rows, 3);
    mat x2(data.n_rows, 3);
    
    for (size_t i = 0; i < 3; ++i) 
        x1.col(i) = data.col(i);
    for (size_t i = 5; i >= 3; --i) 
        x2.col(i-3) = data.col(i);

    mat y_tag(data.n_rows, 1);
    y_tag.col(0) = data.col(6);
    
    mat data_for_disc(2*data.n_rows, 3);
    
    for (size_t j = 0; j < x1.n_rows; ++j) 
        data_for_disc.row(j) = x1.row(j);
    for (size_t j = x2.n_rows; j < 2*(x2.n_rows); ++j) 
        data_for_disc.row(j) = x2.row(j-x2.n_rows);
        
    std::vector<short> tags(data.n_rows, 0);
    for (size_t i = 0; i < tags.size(); ++i) 
        tags[i] = y_tag[i] != 1 ? -1 : 1;
    
    mat dis_points = u.find_discrit_points(data_for_disc,5);
    stdvecvec dis_points2 = u.mat_to_std_vec_2(dis_points);
    std::vector<std::vector<double> > examples = u.mat_to_std_vec(data_for_disc);

	std::vector<std::vector<size_t> > training;				   
	std::vector<short> training_tags;
	std::vector<std::vector<size_t> > testing;
	std::vector<short> testing_tags;
	
	id_devide.dividePrecetage(examples, tags, training, training_tags, testing, testing_tags);
	
	std::cout << " **** after division ****\n" << std::endl;
    for(uword row=0; row < dis_points.n_rows; ++row){
        for(uword col=0; col < dis_points.n_cols; ++col){
            cout << dis_points(row,col) << ' '; 
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
        
        learn.SGD(examples, training, training_tags, dis_points2, indices_of_groups,
                   id_pair, if_equal_dist_zero, if_non_equal_dist_non_zero, 
                   NON_EQUAL_EPSILON, symmetry, Wreg, C[i], W, thold);
                   
        for (size_t j = 0; j < testing.size(); j++) {
            std::vector<Pair> vol = id_pair( examples[ testing[j][0] ], examples[ testing[j][1] ] );
            short s = learn.classification(W, vol, thold);
    
            if(s != testing_tags[j]) numOfErrors++;
        }
        
        std::cout << "C[" << i << "] = " << C[i] << std::endl; 
        std::cout << "num of errors: " << numOfErrors << std::endl;
        std::cout << "errors precent: " << (double)numOfErrors/testing_tags.size() << std::endl; 
        std::cout << "thold: " << thold << "\n\n" << std::endl;
        numOfErrors = 0;
        thold = 1;
        
        // learn.printvec(W);          ///////////////////////////
    }
	
	
    /*std::vector< std::vector<size_t> > comb = create_combination_2(10);
    for (auto i : comb) {
        for (auto j : i) {
            std::cout << j << " ";
        }std::cout << std::endl;
    }*/
    /*    std::size_t n = 6;
    std::size_t k = 2;
    
    std::vector<int> ints;
    for (int i = 0; i < n; ints.push_back(i++));
    
    do
    {
       for (int i = 0; i < k; ++i)
       {
          std::cout << ints[i] << " ";
       }
       std::cout << "\n";
    }
    while(u.next_combination(ints.begin(),ints.begin() + k,ints.end()));*/
    /*
        0 1 
        0 2 
        0 3 
        0 4 
        0 5 
        1 2 
        1 3 
        1 4 
        1 5 
        2 3 
        2 4 
        2 5 
        3 4 
        3 5 
        4 5 
    */
    /*std::vector<short> myvector;
    
    // set some values:
    for (short i=1; i<10; i++) myvector.push_back(i);               // 1 2 3 4 5 6 7 8 9
    
    std::replace_if (myvector.begin(), myvector.end(), IsOdd, -1); // 0 2 0 4 0 6 0 8 0
    
    std::cout << "myvector contains:";
    for (std::vector<short>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << '\n';*/
    
    return 0;
}