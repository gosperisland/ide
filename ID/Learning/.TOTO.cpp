#include <cstdlib>
#include <iostream>
#include <ctime>
#include <experimental/random>

using namespace std;

class Functor{
public:
    
    void operator()(double c, double f){
        cout<<"c -"<<c<<endl;
        cout<<"f -"<<f<<endl;
    }
};

int main(){
    // Functor f;
    // Functor::operator()(4,5);
    // static Functor::operator()(4,5);
    
    srand(time(NULL));
    for (int i = 0; i < 10; i++) 
        std::cout << rand() % 10 << " ";
    std::cout << std::endl;
    
        std::srand(std::time(0)); // use current time as seed for random generator
    for (int i = 0; i < 10; i++){ 
        std::cout << std::experimental::randint(0, 10) << " ";
    }std::cout << std::endl;
    
    return 1;
}