#include <iostream>
#define ZERO 0
#include <climits>

int main()
{
    using namespace std;
    short Dohee = SHRT_MAX;
    unsigned short Insuk = Dohee;

    cout << "in Dohee's account " << Dohee << " won is in there, ";
    cout << "in Insuk's accout " << Insuk << " won is in there." << endl;
    cout << "각각의 계좌에 1원씩 입금한다." << endl << "이제 ";
    Dohee = Dohee + 1;
    Insuk = Insuk + 1;
    cout << "in Dohee's account " << Dohee << " won is in now, ";
    cout << "in Insuk's accout " << Insuk << " won is in now." << endl;
    cout << "oh my! 도희가 나몰래 대출을 했나?" << endl;
    Dohee = ZERO;
    Insuk = ZERO;
    cout << "in Dohee's account " << Dohee << " won is in there, ";
    cout << "in Insuk's accout " << Insuk << " won is in there." << endl;
    cout << "각각의 계좌에서 1원씩 인출한다." << endl << "이제 ";
    Dohee = Dohee - 1;
    Insuk = Insuk - 1;
    cout << "in Dohee's account " << Dohee << " won is in now, ";
    cout << "in Insuk's accout " << Insuk << " won is in now." << endl;
    cout << "oh my! 인숙이가 복권에 당첨되었나?" << endl;
    return 0;
}