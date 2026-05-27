#include <iostream>

// and inline function definition
inline double square(double x){ return x * x;}      // ???

int main()
{
    using namespace std;
    double a, b;
    double c = 13.0;

    a = square(5.0);
}