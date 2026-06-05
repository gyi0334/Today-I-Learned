#include <iostream>
#include <cctype>   // or ctype.h
#include "stack.h"
int main()
{
    using namespace std;
    Stack st;   // create an empty stack
    char ch;
    unsigned long po;
    cout << "Please enter A to add a purchase order,\n"
        << "P to process a PO, or Q to quit.\n";
    while (cin >> ch && toupper(ch) != 'Q')     // toupper : 소문자를 대문자로
    {
        while (cin.get() != '\n')
            continue;
        if (!isalpha(ch))
        {
            cout << '\a';   // alam 삑.
            continue;
        }
        switch(ch)
        {
            case 'A':
            case 'a': cout << "Enter a PO number to add: ";
                    cin >> po;
                    if (st.isfull())
                        cout << "stack already full\n";
                    else
                        st.push(po);
                    break;
            case 'P':
            case 'p': if (st.isempty())
                        cout << "stack arlready empty\n";
                    else {
                        st.pop(po);
                        cout << "PO #" << po << " popped\n";
                    }
                    break;
        }
        cout << "Please enter A to add a purchase order,\n"
            << "P to process a PO, or Q to quit.\n";
    }
    cout << "Bye\n";
    return 0;
}