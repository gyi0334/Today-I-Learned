#include <iostream>
int main()
{
	using namespace std;
	char ch = 'M';
	int i = ch;
	cout << ch << "의 ASCII 코드는 " << i << " 입니다." << endl;

	cout << "이 문자 코드에 1을 더해 보겠습니다." << endl;
	ch = ch + 1;
	i = ch;
	cout << ch << "의  ASCII 코드는 " << i << " 입니다." << endl;

	cout << "using cout.put(ch), print variable ch (type char) : ";
	cout.put(ch);

	cout.put('!');

	cout << endl << "END." << endl;
	return 0;
}
