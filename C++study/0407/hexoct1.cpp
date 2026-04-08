#include <iostream>
int main()
{
    using namespace std;
    int chest = 42;    // 10진 정수형 상수
    int waist = 0x18;  // 16진 정수형 상수
    int inseam = 042;  // 8진 정수형 상수

    cout << "가슴둘레 " << chest << "\n";
    cout << "허리가 몇이니? " << waist << "요." << "\n";
    cout << "힙은? " << inseam << "요." << "\n";
    cout << "와우! 어머님이 누구니" << endl;
    return 0;
}