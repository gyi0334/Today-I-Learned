#include <iostream>
#include <string>
int main()
{
    using namespace std;
    cout << "Enter a word: ";
    string word;
    cin >> word;    // stressed

    //physically modify string object
    char temp;
    int i, j;
    for (j = 0, i = word.size() - 1; j < i; --i, ++j)
    {
        cout << "i = " << i << "j = " << j << endl;
        temp = word[i];     // temp = d 임시 저장
        word[i] = word[j];  // word[7] = word[0] 맨 뒤로 보내기
        word[j] = temp;     // word[0] = temp = d 맨 앞으로 보내기
    }
    cout << word << "\nDone\n";
    return 0;
}