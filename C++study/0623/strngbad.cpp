#include <cstring>
#include "strngbad.h"
using std::cout;

int StringBad::num_strings = 0;

// construct StringBad from C string
StringBad::StringBad(const char * s)
{
    len = std::strlen(s);
    str = new char[len + 1];
    std::strcpy(str, s);
    num_strings++;
    cout << num_strings << ": \"" << str << "\" object created\n";
}

StringBad::StringBad()
{
    len = 4;
    str = new char[4];
    std::strcpy(str, "C++");
    num_strings++;
    cout << num_strings << ": \"" << str
        << "\" default object created\n";   // FYI
}

StringBad::~StringBad()     // necessary destructor
{
    cout << "\"" << str << "\" object deleted, ";
    --num_strings;          // ??
    cout << num_strings << " left\n";
    delete [] str;          // str을 쓰고나서 바로 delete 해도 되는거 아닌가
}

std::ostream & operator << (std::ostream & os, const StringBad & st)
{
    os << st.str;
    return os;
}