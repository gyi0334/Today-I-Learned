#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::cout << "Hello OpenCV" << CV_VERSION<< std::endl;

    cv::Mat img;
    img = cv::imread("lenna.bmp");

    if (img.empty())
    {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    cv::namedWindow("image");
    cv::imshow("image",img);

    cv::waitKey();
    return 0;
}