// structure template
#ifndef COORDIN_H_      // if not defined(ifndef)
#define COORDIN_H_

struct polar
{
    double distance;    // distance from origin
    double angle;       // direction from origin
};

struct rect
{
    double x;           // horizontal distance from origin
    double y;           // vertical distance from origin
};

// prototypes
polar rect_to_polar(rect xypos);
void show_polar(polar dapos);

#endif