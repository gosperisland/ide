#ifndef _SGD_LEARNING_H
#define _SGD_LEARNING_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cassert>
#include <exception>
#include <time.h>
#include "armadillo"
#include "./ID/IDpair.hpp"




using namespace std;
using namespace arma;

#define EPOCH_TIMES 3
static int prunePairsFactor = 1;
static double thresholdValue = 1;

template <typename T> void print(ostream& os, const vector<T> arg)
{
	for (size_t i=0;i<arg.size(); i++){
		os << " " << arg[i] ;
	}
}


template <typename T> ostream& operator<<(ostream& os, vector<T>& arg)
{
    print(os, arg);
    return os;
}

void printFlattenedSquare(const vector<double> & W){
	cout << "W.size(): " << W.size() << " \nW:" << endl;
	for (size_t i = 0; i < W.size(); i++) {
		cout << W[i] << " , ";
		if ((i + 1) % (int) sqrt(W.size()) == 0) {
			cout << endl;
		}
	}
}


class SGDLearning{
public:
virtual ~SGDLearning(){

}


protected:

std::vector<double> construct_Wreg(Grid points_pair) {
	points_pair.double_grid();
	size_t num_of_ver = points_pair.get_num_of_vertices();

	std::vector<double> Wreg(num_of_ver, 0);
	for (size_t i = 0; i < num_of_ver; i++) {
		Wreg[i] = thresholdValue;
	}
	return Wreg;
}

inline vector<int> indToPair(int arg, int size){
	int i = arg / size;//line
	int j = arg % size;//row
	return {i,j};
}

inline int pairToInd(int i, int j, int n){
	return n * i + j;
}

virtual void makeSymmetric(std::vector<double> & W) {
	float eps = 0;
	int n = (int) sqrt(W.size());
	if (abs(sqrt(W.size()) - n) > eps) {
		cerr << "makeSymmetric: flat vector cannot represent square matrix" << endl;
		exit(1);
	}
	for (size_t ind = 0; ind < W.size(); ind++) {
		vector<int> v = indToPair(ind,n);
		int i=v[0];
		int j=v[1];
		if (j > i) {// in the source pair-index
			W[pairToInd(j,i,n)] = W[ind];//target pair-index is swapped
		}

	}
}



virtual std::vector<double> learn_similar(
		const std::vector<std::vector<double> >& examples, const IDpair& idpair,
		const std::vector<std::vector<size_t> >& indices_of_pairs,
		const std::vector<short>& tags, const std::vector<double>& Wreg,
		const double C, double& thold) = 0;


public:
std::vector<double> run(const std::vector<std::vector<double> >& examples,
		const std::vector<std::vector<size_t> >& indices_of_pairs,
		const std::vector<short>& tags,
		const std::vector<std::vector<double> >& discrete_points,
		const double C, double& thold) {
	std::vector<double> Wreg;
	std::vector<double> W;

	Grid grid_pair(discrete_points);

	IDpair id_pair(grid_pair);

	Wreg = construct_Wreg(grid_pair);

	W = learn_similar(examples, id_pair, indices_of_pairs, tags, Wreg, C,
			thold);

	return W;
}



// -1(similar) , 1(non-similar)
short classification(const std::vector<double>& W,
		const std::vector<Pair>& non_zero, double thold) {
	double dotProd = 0;
	short ans = 0;

	for (size_t i = 0; i < non_zero.size(); i++)
		dotProd += non_zero[i]._weight * W[non_zero[i]._index];

	if ((dotProd - thold) < 0)
		ans = -1;
	else
		ans = 1;

	return ans;
}

};

#endif
