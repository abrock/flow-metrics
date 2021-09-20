#ifndef COLORS_HPP
#define COLORS_HPP

#include <opencv2/highgui.hpp>

struct ColorsRD {
    static cv::Vec3b red() {return cv::Vec3b(65,65,236);}
    static cv::Vec3b orange() {return cv::Vec3b(53,113,239);}
    static cv::Vec3b yellow() {return cv::Vec3b(95,232,253);}
    static cv::Vec3b green() {return cv::Vec3b(78, 187, 95);}
    static cv::Vec3b purple() {return cv::Vec3b(134, 46, 99);}
    static cv::Vec3b blue() {return cv::Vec3b(209, 152, 27);}
    static cv::Vec3b lightBlue() {return cv::Vec3b(245, 219, 155);}
};

#endif // COLORS_HPP
