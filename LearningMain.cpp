//

#include "SGDLearning.hpp"
#include "NormalSGDLearning.hpp"
#include <ctime>


double L1DistanceScalar(double p1, double p2){
  return fabs(p1-p2);
}

double L2Distance(vec p1, vec p2){
  vec diff = (p1-p2);
  double res =  sqrt(dot(diff,diff));
  return res;
}




vvoid sgdMain() {
	int dim = 1;//default value
	bool useJustFromGrid = false; //taking samples from the grid itself
	bool useUnifiedGrid =true;
	bool doPrintW = false;
	thresholdValue = 40;
	vec rep = thresholdValue * ones(dim,1);
	thresholdValue = sqrt(dot(rep,rep));
	cout<<"Dimension:"<<dim<<endl;
	cout<<"thresholdValue: "<<thresholdValue<<endl;


	if (useJustFromGrid && !useUnifiedGrid){
		cerr<<"sgdMain (useJustFromGrid && !useUnifiedGrid) not supported.";
		exit(1);
	}
	const size_t numOfSamples = 5000000; //5M ~217sec
	//const size_t numOfSamples = 500000; //500k ~22sec
	//const size_t numOfSamples = 250000;
	//const size_t numOfSamples = 50000; //50k ~2sec
	//const size_t numOfSamples = 500;

	std::vector<std::vector<double> > discrete_points;
	std::vector<double> gridForX1;
	if (useUnifiedGrid){//unified for all dimensions of the feature vectors

		gridForX1 =
				{ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

		for (int d=0; d< dim; d++){
			discrete_points.push_back(gridForX1);
		}
	}
	else{//specify each grid separately

		gridForX1 =
						{ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

		std::vector<double> gridForX2 = { 0, 200 };
		discrete_points.push_back(gridForX1);
		discrete_points.push_back(gridForX2);
		dim = discrete_points.size();
	}
	//std::vector<std::vector<double> > discrete_points = {{0,0},{0,200},{100,0},{100,200}};
	std::vector<vector<double> > examples;
	std::vector<std::vector<size_t> > indices_of_pairs;
	//arma_rng::set_seed_random();
	if (useJustFromGrid) {
		for (size_t j = 0; j < gridForX1.size(); j++) {
			vector<double> cur_vec;
			for (int d = 0; d < dim; d++)
				cur_vec.push_back(discrete_points[d][j]);
			examples.push_back( cur_vec );
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
			vector< vec > A;
			for (int d = 0; d < dim; d++)
				A.push_back(randi<vec>(numOfSamples / 2, distr_param(0, 100)));

			for (size_t j = 0; j < A[0].size(); j++) {
				vector<double> cur_vec;
				for (int d = 0; d < dim; d++)
					cur_vec.push_back( A[d](j) );
				examples.push_back(cur_vec);

			}

		}
		for (size_t i = 0; i < numOfSamples / 2; i++) {
			vector<size_t> cur_pair = { i, i + numOfSamples / 2 }; //0000001000,0000001000
			indices_of_pairs.push_back(cur_pair);
		}
	}
	vector<short> tags(indices_of_pairs.size(), 0);
	// generate tags.
	int counter = 0;
	for (size_t i = 0; i < indices_of_pairs.size(); i++) {

		double dist = L2Distance(examples[indices_of_pairs[i][0]],
				examples[indices_of_pairs[i][1]]);

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
	SGDLearning * learning = createSGDLearning();
	std::vector<double> W = learning->run(examples, indices_of_pairs, tags, gridpair, 0.5,
			tholdArg);


	if (doPrintW){
		//Flattened
		printFlattenedSquare(W);
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
	sgdMain();
	return 0;
}
