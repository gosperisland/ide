#ifndef _REGULARIZATION
#define _REGULARIZATION

class Regularization{
public:
    Regularization() {}

    std::vector<double> construct_Wreg(Grid& points_pair){
        std::vector<double> Wreg(points_pair.get_num_of_vertices(), 0);
        std::vector<double> pair_vertex(points_pair.get_num_of_dims(), 0);
        std::vector<double> first_vertex(pair_vertex.size()/2, 0);
        std::vector<double> second_vertex(pair_vertex.size()/2, 0);

        for (size_t i = 0; i < points_pair.get_num_of_vertices() - 1; i++) {
            
            //get a vertex from the grid by index
            points_pair.get_vertex(i, pair_vertex);
            std::vector<double>::iterator it = pair_vertex.begin();
            
            //the first half of pair_vertex holds the values of the first point
            first_vertex.assign(it , it + pair_vertex.size()/2);
            it += pair_vertex.size()/2;

            //the second half of pair_vertex holds the values of the second point
            second_vertex.assign(it , it + pair_vertex.size()/2);
            
            //assign Wreg the value (|second_vertex|-|first_vertex|)
            Wreg[i] = L1Distance(first_vertex, second_vertex);
        }

        return Wreg;
    }

    double L2Distance(const std::vector <double>& p1, const std::vector <double>& p2){
        assert(p1.size() == p2.size());
        double dist = 0;
        std::vector <double> difference(p1.size(), 0);

        for (size_t i = 0; i < p1.size(); i++) {
            double x = p2[i] - p1[i];
            difference[i] = std::pow(x,2);
            dist += difference[i];
        }

        return std::sqrt(dist);
    }

    double L1Distance(const std::vector <double>& p1, const std::vector <double>& p2){
        assert((p1.size() == p2.size()) && "L1Distance :: expected equal size vectors");
        double dist = 0;

        for (size_t i = 0; i < p1.size(); i++)
            dist += fabs( p2[i] - p1[i] );  
        
        return dist;
    }

    double L1DistanceScalar(double p1, double p2){
        return fabs(p1 - p2);
    }
    
    std::vector<double> c_vec_intialization() {
        std::vector<double> c_vec;
        
        for (int n = -10; n <= 10; n += 2)
            c_vec.push_back(std::pow(2, n));
            
        return c_vec;
    }

/*    std::vector<double> c_vec_intialization() {
        std::vector<double> c_vec;
        
        for (int n = 150; n <= 500; n += 50)
            c_vec.push_back(n);
            
        return c_vec;
    }
*/    
};

#endif
