#include <iostream>
#include <climits>  // 구식 C++에서는 limits.h를 사용한다.
int main(void)
{
 using namespace std;
 int n_int = INT_MAX;
 short n_short = SHRT_MAX;
 long n_long = LONG_MAX;
 long long n_llong = LLONG_MAX;

 // sizeof 연산자는 데이터형이나 변수의 크기를 알라낸다.
 cout << "int is... " << sizeof (int) << "byte." << endl;
 cout << "short is..." <<sizeof n_short << "byte." << endl;
 cout << "long is..." <<sizeof n_long << "byte." << endl;
 cout << "longlong is..." <<sizeof n_llong << "byte." << endl;
 cout << endl;

 cout << "max : " << endl;
 cout << "int : " << n_int << endl;
 cout << "short : " << n_short << endl;
 cout << "long : " << n_long << endl << endl;
 cout << "long long : " << n_llong << endl << endl;
 cout << "min of int : " << INT_MIN << endl;
 cout << "number of bit/byte : " << CHAR_BIT << endl;
 return 0;
}
