#include <iostream>
#include <cmath>
#include "coordin.h"    // structure templates, function prototypes

//convert rectagular to polar coordinates
polar rect_to_polar(rect xypos)
{
    using namespace std;
    polar answer;

    answer.distance = sqrt(xypos.x * xypos.x + xypos.y * xypos.y);
    answer.angle = atan2(xypos.y, xypos.x);
    return answer;      // returns a polar structure
}

// show polar coordinates, converting angele to degrees
void show_polar(polar dapos)
{
    using namespace std;
    const double Rad_to_deg = 57.29577951;

    cout << "distance = " << dapos.distance;
    cout << ", andge = " << dapos.angle * Rad_to_deg;
    cout << " degrees\n";
}

// if 실행시키고 싶으면 g++ file1.cpp file2.cpp