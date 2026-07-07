#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void MatOp5()
{
    Mat img1 = imread("lenna.bmp");

    cout << "Width: " << img1.cols << endl;
    cout << "Height: " << img1.rows << endl;
    cout << "Channels: " << img1.channels() << endl;

    if(img1.type() == CV_8UC1)
        cout << "img5 is a grayscale image." << endl;
    else if(img1.type() == CV_8UC3)
        cout << "img5 is a truecolor image." << endl;

    float data[] = {2.f, 1.414f, 3.f, 1.732f};
    Mat mat1(2,2, CV_32FC1, data);
    cout << "mat1:\n" << mat1 << endl;
}

void MatOp6()
{
    float data[] = {1,1,2,3};
    Mat mat1(2,2,CV_32FC1, data);
    cout << "mat1:\n" << mat1 << endl;

    Mat mat2 = mat1.inv();
    cout << "mat2:\n" << mat2 << endl;

    cout << "mat1.t():\n" << mat1.t() << endl;
    cout << "mat1 + 3:\n" << mat1 + 3 << endl;

    cout << "mat1 + mat2:\n" << mat1 + mat2 << endl;
    cout << "mat1 * mat2:\n" << mat1 * mat2 << endl;
}

int main()
{
    MatOp5();
    MatOp6();
    return 0;
}