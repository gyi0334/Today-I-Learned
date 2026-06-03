// version 00
#ifndef STOCK00_H_
#define STOCK00_H_

#include <string>

class Stock // class declaration
{
    private:
        std::string company;    // 회사명
        long shares;            // 보유 주식 수
        double share_val;       // 주식 1주의 가치
        double total_val;       // 보유 주식의 총 가치
        void set_tot() {total_val = shares * share_val;}
    public:
        void acquire(const std::string & co, long n, double pr);
        void buy(long num, double price);
        void sell(long num, double price);
        void update(double price);
        void show();
};

#endif
