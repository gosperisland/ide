#ifndef _UTILS
#define _UTILS

#include "armadillo"

class Utils{
public:
    std::vector<short> _tags;
    std::vector<std::vector<double> > _examples;
    std::vector<std::vector<double> > _discrete_points_for_all_dims;
    
    Utils():
    _tags(1,0),
    _examples(0, std::vector<double>(0)),
    _discrete_points_for_all_dims(0, std::vector<double>(0)) {}

    void readCSV(const std::string& filename){
        arma::mat data = loadFileToMatrix(filename);

        arma::mat x1(data.n_rows, 3);
        arma::mat x2(data.n_rows, 3);
        
        for (size_t i = 0; i < 3; ++i) 
            x1.col(i) = data.col(i);
        for (size_t i = 5; i >= 3; --i) 
            x2.col(i-3) = data.col(i);
            
        arma::mat labels(data.n_rows, 1);
        labels.col(0) = data.col(6);
        
        convertLables(labels);

        arma::mat examples(2*data.n_rows, 3);
        for (size_t j = 0; j < x1.n_rows; ++j) 
            examples.row(j) = x1.row(j);
        for (size_t j = x2.n_rows; j < 2*(x2.n_rows); ++j) 
            examples.row(j) = x2.row(j-x2.n_rows);
            
        arma::mat _discrete_points = findDiscretePoints(examples, 2);
        _discrete_points_for_all_dims = matToVecByColumns(_discrete_points);
        _examples = matToVecByRows(examples);
    }
    
    //get filename return matrix of data                                    //add param &data **************************
    arma::mat loadFileToMatrix(const std::string& filename){
        arma::mat data;
        data.load(filename,arma::csv_ascii);
        return data;
    }

    template <typename T>
    void convertLables(const arma::Mat<T>& labels){
        _tags.resize(labels.size());
        for (size_t i = 0; i < _tags.size(); i++)
            _tags[i] = labels[i] != 1 ? -1 : 1;
    }
    
    std::vector< std::vector<double> > matToVecByRows(const arma::mat& A){
        std::vector< std::vector<double> > V(A.n_rows);

        for (size_t i = 0; i < A.n_rows; ++i) 
            V[i] = arma::conv_to< std::vector<double> >::from(A.row(i));
        
        return V;
    }

    std::vector< std::vector<double> > matToVecByColumns(arma::mat& A){
        std::vector< std::vector<double> > V(A.n_cols);

        for (size_t i = 0; i < A.n_cols; ++i)
            V[i] = arma::conv_to< std::vector<double> >::from(A.col(i));
        
        return V;
    }
    
    std::vector< std::vector<double> > transposeMatToVec(const arma::mat& A){
        std::vector< std::vector<double> > V(A.n_cols);

        for (size_t i = 0; i < A.n_cols; ++i) 
            V[i] = arma::conv_to< std::vector<double> >::from(A.col(i));
        
        return V;
    }   //not in use!!! check with Ofir  **************************
    
    /* generate n choose k combination of indexes */                        //not in use!!! check with Ofir  **************************
	template <typename Iterator>    
    inline bool nextCombination(const Iterator first, Iterator k, const Iterator last){
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
    
    /*
    get matrix of data and number of chunks
    return matrix of discrit point for each dim.
    */
    arma::mat findDiscretePoints(const arma::mat& data, const size_t& number_of_chunks){
        arma::mat disc_point(number_of_chunks, data.n_cols, arma::fill::zeros);
        //getting K parts by K-1 sections
        int chunk_size = data.n_rows/(number_of_chunks-1); 
        for (size_t i = 0; i < data.n_cols; ++i) {
            //extract vector and sort it.
            arma::vec z = sort(data.col(i));
            //ignore last step .
            for (size_t j = 0; j < number_of_chunks-1; ++j) 
                disc_point.col(i)[j]= z[j*chunk_size];

            disc_point.col(i)[number_of_chunks-1]= z[data.n_rows-1];
        }
        return disc_point;
    }

    std::vector< std::vector<size_t> > createPairCombinations(size_t num_rows){
        //generate vector
        std::vector<size_t> indexes(num_rows);
        //fill with indexes
    
        std::vector< std::vector<size_t> > _combinations;
    
        for (size_t j = 0; j < num_rows; ++j) {
            std::vector<size_t> _tempVector{j, j+num_rows};
    
            _combinations.push_back(_tempVector);
        }
    
        return _combinations;
    }
};

#endif
