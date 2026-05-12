#include <iostream>
int main()
{
    char ch;

    std::cout << "Type, and I shall repeat.\n";
    std::cin.get(ch);
    while (ch != '.')
    {
        if (ch == '\n')
            std::cout << ch;        // done if newline
        else
            std::cout << ++ch;      // done otherwise
        std::cin.get(ch);
    }
    // try ch + 1 instead of ++ch for intersting effect
    std::cout << "\nPlease excuse the slight confusion.\n";
        // std::cin.get();
        // std::cin.get();
    return 0;
}