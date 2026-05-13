#include <iostream>
#include <cctype>

int main()
{
    using namespace std;
    cout << "Enter text for analysis, and type @ to terminate input.\n";
    char ch;
    int whitespace = 0;
    int digits = 0;
    int chars = 0;
    int punct = 0;
    int others = 0;

    cin.get(ch);                // get first character
    while (ch != '@')           // test for sentinel
    {
        if(isalpha(ch))         // is it an alphabetic character?
            chars++;
        else if (isspace(ch))   // is it a whitespace character?
            digits++;
        else if (ispunct(ch))
            punct++;
        else
            others++;
        cin.get(ch);
        cout << chars << " letters, "
            << whitespace << " whitespace, "
            << digits << " digits, "
            << punct << " punctuatuations, "
            << others << " others.\n";
        return 0;
    }
}