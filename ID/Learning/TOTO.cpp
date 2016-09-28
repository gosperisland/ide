#include <iostream>
using namespace std;

class Functor{
public:
    
    static void operator()(double c, double f){
        cout<<"c -"<<c<<endl;
        cout<<"f -"<<f<<endl;
    }
};

int main(){
    // Functor f;
    // Functor::operator()(4,5);
    static Functor::operator()(4,5);
    return 1;
}