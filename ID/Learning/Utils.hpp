#ifndef _UTILS
#define _UTILS

#include <armadillo>

class Utils{
public:
    
    //get filename return matrix of data
    arma::mat load_file_to_matrix(const std::string& filename){
        arma::mat data;
        data.load(filename,arma::csv_ascii);
        return data;
    }

    std::vector< std::vector<double> > mat_to_std_vec(const arma::mat& A){
        std::vector< std::vector<double> > V(A.n_rows);

        for (size_t i = 0; i < A.n_rows; ++i) 
            V[i] = arma::conv_to< std::vector<double> >::from(A.row(i));
        
        return V;
    }

    std::vector< std::vector<double> > transpose_mat_to_std_vec(const arma::mat& A){
        std::vector< std::vector<double> > V(A.n_cols);

        for (size_t i = 0; i < A.n_cols; ++i) 
            V[i] = arma::conv_to< std::vector<double> >::from(A.col(i));
        
        return V;
    }
    
    // generate n choose k combination of indexes
	template <typename Iterator>
    inline bool next_combination(const Iterator first, Iterator k, const Iterator last){
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
    
    std::vector< std::vector<double> > mat_to_std_vec_2(arma::mat &A){
        std::vector< std::vector<double> > V(A.n_cols);

        for (size_t i = 0; i < A.n_cols; ++i)
            V[i] = arma::conv_to< std::vector<double> >::from(A.col(i));
        
        return V;
    }

    /*
    get matrix of data and number of chunks
    return matrix of discrit point for each dim.
    */
    arma::mat find_discrit_points(const arma::mat & data, const size_t & number_of_chunks){
        arma::mat disc_point(number_of_chunks, data.n_cols, arma::fill::zeros);
        //-1 becuase for to get K parts we need to do K-1 "Cuts"
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

    std::vector< std::vector<size_t> > create_combination_2(size_t num_rows){
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
