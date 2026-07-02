#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "Camera open failed" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        cv::imshow("Webcam", frame);

        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}