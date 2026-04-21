#include <iostream>
struct inflatable
{
    char name[20];
    float volume;
    double price;
};
int main()
{
    using namespace std;
    inflatable * ps = new inflatable; // allot memory for structure inflatable 이 자료형이었어?
    cout <<"Enter name of inflatable item: ";
    cin.get(ps->name, 20);            // mothod 1 for member access  ps->name 뭔데?
    cout << "Enter volume in cubic feet: ";
    cin >> (*ps).volume;              // mothod 2 for member access  이건 또 뭐여?
    cout << "Enter price: $";
    cin >> ps->price;
    cout << "Name: " << (*ps).name << endl;               // method 2
    cout << "Volume: " << ps->volume << " cubic feet\n";  // mothod 1
    cout << "Price: $" << ps->price << endl;              // mothod 1
    delete ps;                        // free memory used by structure.
    return 0;
}