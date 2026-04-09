#include <iostream>
int main()
{
    using namespace std;
    cout.setf(ios_base::fixed, ios_base::floatfield);
    cout << "정수 나눗셈 : 9/5 = " << 9/5 <<  endl;
    cout << "부동 소수점수 나눗셈 : 9.0/5.0 = ";
    cout << 9.0/5.0 << endl;
}