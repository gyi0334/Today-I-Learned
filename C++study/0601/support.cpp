#include <iostream>
extern double warming;

void update(double dt);
void local();

using std::cout;
void update(double dt)
{
    extern double warming;
    warming += dt;      // uses global warming
    cout << "Updating global warming to " << warming;
    cout << " degrees.\n";
}

void local()
{
    double warming = 0.8;

    cout << "LOCAL warming = " << warming << " degrees.\n";
        // access global variable with the scope resolution poerator
    cout << "but global warming = " << ::warming;
    cout << " degrees.\n";
}