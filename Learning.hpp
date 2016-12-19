#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cassert>
#include <exception>
#include <time.h>
#include "armadillo"
#include "./ID/IDpair.hpp"
//#include "Learning.hpp"
//#include "GridGroup.hpp"


using namespace std;
using namespace arma;

#define EPOCH_TIMES 3
static int prunePairsFactor = 1;
static double thresholdValue = 1;

template <typename T> void print(ostream& os, const vector<T> arg)
{
	for (size_t i=0;i<arg.size(); i++){
		cout << " " << arg[i] ;
	}
}

template <typename T> ostream& operator<<(ostream& os, vector<T>& arg)
{
    print(os, arg);
    return os;
}




class Learning{
private:

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

void makeSymmetric(std::vector<double> & W) {
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



void SGD_similar(std::vector<double>& W, const std::vector<double>& Wreg,
		const std::vector<Pair>& volume, short tag, double & thold,
		const double C, size_t etha) {

	std::vector<double> W_old(W);
	size_t size = W.size();
	size_t sparse_size = volume.size();
	double dotProd = 0;


	//ID(X_pi_1, X_pi_2) * W
	for (size_t i = 0; i < sparse_size; i++)
		dotProd += volume[i]._weight * W[volume[i]._index];

	// 1 - { (ID(X_pi_1, X_pi_2) * W) - threshold } * y_i
	if ((1 - ((dotProd - thold) * tag)) > 0) {
		for (auto& simplex_point : volume) {

			W[simplex_point._index] += (1.0 / (double) etha)
					* (C * tag * simplex_point._weight);
			W[simplex_point._index] =
					W[simplex_point._index] < 0 ? 0 : W[simplex_point._index];
		}
		//thold -= ((1.0 / (double) etha) * (C * tag));

	}

	for (size_t i = 0; i < size; i++) {
		W[i] -= (1.0 / (double) etha) * ((W_old[i] - Wreg[i]));
		W[i] = W[i] < 0 ? 0 : W[i];

	}


}
//indices
std::vector<double> learn_similar(
		const std::vector<std::vector<double> >& examples, const IDpair& idpair,
		const std::vector<std::vector<size_t> >& indices_of_pairs,
		const std::vector<short>& tags, const std::vector<double>& Wreg,
		const double C, double& thold) {

	assert(tags.size() == indices_of_pairs.size());

	size_t W_size = idpair.get_total_num_of_vertices();
	//cout<<Wreg.size()<<","<< W_size;

	std::vector<double> W(Wreg.size(), 0);

	assert(Wreg.size() == W_size);

	size_t num_of_pairs = indices_of_pairs.size();

	bool isRandomInd = true;
	for (int j = 0; j < EPOCH_TIMES; ++j) {

		std::vector<int> random_indexes(num_of_pairs);
		std::iota(std::begin(random_indexes), std::end(random_indexes), 0); // Fill with 0, 1, ..., n.
		std::random_shuffle(random_indexes.begin(), random_indexes.end());

		for (size_t i = 0; i < num_of_pairs / prunePairsFactor; i++) {

			//get random index
			size_t random_index = isRandomInd ? random_indexes.back() : i;

			random_indexes.pop_back();

			const std::vector<Pair>& volume = idpair(
					examples[indices_of_pairs[random_index][0]],
					examples[indices_of_pairs[random_index][1]]);

			SGD_similar(W, Wreg, volume, tags[random_index], thold, C, i + 1);
		}

	}
	makeSymmetric(W);
	return W;
}




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



// 1(similar) , -1(non-similar)
short classification(const std::vector<double>& W,
		const std::vector<Pair>& non_zero, double thold) {
	double dotProd = 0;
	short ans = 0;

	for (size_t i = 0; i < non_zero.size(); i++)
		dotProd += non_zero[i]._weight * W[non_zero[i]._index];

	if ((dotProd - thold) < 0)
		ans = 1;
	else
		ans = -1;

	return ans;
}

};

