#ifndef STNEWT_H_
#define STNEWT_H_
class Stonewt
{
    private:
        enum{Lbs_per_stn = 14};         // pounds per stone
        int stone;                      // whole stones
        double pds_left;                // fractional pounds
        double pounds;                  // entire weight in pounds
    public:
        Stonewt(double lbs);            // constructor for double punds
        Stonewt(int stn, double lbs);   // constructor for stone, lbs
        Stonewt();                      // default constructor
        ~Stonewt();
        void show_lbs() const;          // show weight in pounds format
        void show_stn() const;          // show weight in stone format
};

#endif