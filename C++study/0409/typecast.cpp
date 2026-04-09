#include <iostream>
int mian()
{
    using namespace std;
    int auks, bats, coots;

    auks = 19.99 + 11.99;

    bats = (int) 19.99 + (int) 11.99;  // 구식 C 스타일
    coots = int (19.00) + int (11.99); // 신식 C++ 스타일
    cout << "바다오리 = "<< auks << ", 박쥐 = " << bats;
    cout << ", 검둥오리 = " << coots << endl;

    char ch = 'Z';
    cout << "코드 " << ch << " 의 값음 ";
    cout << int (ch) << endl;
    cout << "네, 코드 Z 의 값은 ";
    cout << static_cast<int>(ch) << endl;
    return 0;
}