#include "SGDLearning.hpp"
#include "NormalSGDLearning.hpp"
#include <ctime>


double L1DistanceScalar(double p1, double p2){
  return fabs(p1-p2);
}





void sanityTest1Dim() {
	bool justFromGrid = false; //taking samples from the grid itself
	//const size_t numOfSamples = 5000000; //5M ~217sec
	//const size_t numOfSamples = 500000; //500k ~22sec
	const size_t numOfSamples = 50000; //50k ~2sec
	std::vector<double> gridForX1 =
			{ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	//std::vector<double> gridForX1 = {0, 100};
	std::vector<double> gridForX2 = { 0, 200 };
	std::vector<std::vector<double> > discrete_points = { gridForX1 };
	//std::vector<std::vector<double> > discrete_points = {{0,0},{0,200},{100,0},{100,200}};
	std::vector<std::vector<double> > examples;
	std::vector<std::vector<size_t> > indices_of_pairs;
	//arma_rng::set_seed_random();
	if (justFromGrid) {
		for (size_t j = 0; j < gridForX1.size(); j++) {
			examples.push_back( { gridForX1[j] });
		}
		for (size_t i = 0; i < gridForX1.size(); i++) {
			for (size_t j = 0; j < gridForX1.size(); j++) {
				vector<size_t> p1 = { i, j };
				indices_of_pairs.push_back(p1);
			}
		}

	} else {
		// loop for creating pairs
		for (int i = 0; i < 2; i++) {
			vec A = randi<vec>(numOfSamples / 2, distr_param(0, 100));
			vec B = randi<vec>(numOfSamples / 2, distr_param(0, 200));

			for (size_t j = 0; j < A.size(); j++) {
				vector<double> p1 = { A(j) };
				examples.push_back(p1);

			}
		}
		for (size_t i = 0; i < numOfSamples / 2; i++) {
			vector<size_t> p1 = { i, i + numOfSamples / 2 }; //0000001000,0000001000
			indices_of_pairs.push_back(p1);
		}
	}
	vector<short> tags(indices_of_pairs.size(), 0);
	// generate tags.
	int counter = 0;
	for (size_t i = 0; i < indices_of_pairs.size(); i++) {

		double dist = L1DistanceScalar(examples[indices_of_pairs[i][0]][0],
				examples[indices_of_pairs[i][1]][0]);
		thresholdValue = 20;
		tags[i] = dist < thresholdValue ? 1 : -1;

		if (tags[i] < 0)
			counter++;
	}


	cout << "Number of good examples: " << counter << " out of: "
			<< indices_of_pairs.size() << " examples" << endl;

	double tholdArg = thresholdValue;		//argument for init, might not be changed at all in the case of threshold regularization

	std::vector<std::vector<double> > gridpair(discrete_points);
	/*gridpair.insert(gridpair.end(), discrete_points.begin(),
				discrete_points.end());
*/

	time_t tstart, tend;
	tstart = time(0);
	size_t numOfErrors = 0;


	//////for supporting backward compatibility:
#ifdef DOUBLE_OUTSIDE
	gridpair.insert(gridpair.end(), discrete_points.begin(), discrete_points.end());
#endif
	///////////////////
	SGDLearning * learning = new NormalSGDLearning();
	std::vector<double> W = learning->run(examples, indices_of_pairs, tags, gridpair, 2,
			tholdArg);

	bool printW = true;
	if (printW){
		cout << "W.size(): " << W.size() << " \nW:" << endl;
		for (size_t i = 0; i < W.size(); i++) {
			cout << W[i] << " , ";
			if ((i + 1) % (int) sqrt(W.size()) == 0) {
				cout << endl;
			}
		}
	}
	Grid grid(gridpair);
	IDpair id_pair(grid);// is not being used as an input to SGD-init but created inside

	//gridpar vector include discrete_points twice, one for each hyper-axis, i.e., X and Y

	for (size_t i = 0; i < indices_of_pairs.size(); i++) {
		vector<double> examp0 = examples[indices_of_pairs[i][0]];
		vector<double> examp1 = examples[indices_of_pairs[i][1]];

		std::vector<Pair> vol = id_pair(examp0, examp1);

		short s = learning->classification(W, vol, tholdArg);

		if (s != tags[i])
			numOfErrors++;
	}
	delete learning;
	tend = time(0);
	cout << "Number of errors: " << numOfErrors << endl;
	cout << "Error rate: " << (double) numOfErrors / tags.size() << endl;
	//cout << "thold: " << thold << ".\n" << endl;
	cout << "It took " << difftime(tend, tstart) << " second(s)." << endl;
}



int main() {
	sanityTest1Dim();
	return 0;
}
