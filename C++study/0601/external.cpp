// external.cpp -- external variables
// compile with support.cpp
#include <iostream>
using namespace std;
// external variable
double warming = 0.4;       // warming defined
//function prototypes
void update(double dt);
void local();

int main()              // uses global variable
{
    cout << "main Global warming is " << warming << " degrees.\n";
    update(0.1);
    cout << "main Global warming is " << warming << " degrees.\n";
    local();
    cout << "main Global warming is " << warming << " degrees.\n";
    return 0;
}