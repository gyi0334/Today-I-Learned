#include <iostream>
int main()
{
    using namespace std;
    int updates = 6;  // declare a variable
    int * p_updates;  // declare pointer to an int
    p_updates = &updates;

    //express values two ways
    cout << "Values: updates = " << updates;
    cout << " , *p_updates = " << *p_updates << endl;

    //express address two ways
    cout << "Adresses: &updates = " << &updates;
    cout << ", p_updates = " << p_updates << endl;

    //use pointer to change value
    *p_updates = *p_updates + 1;
    cout << "Now updates = " << updates << endl;
    return 0;
}