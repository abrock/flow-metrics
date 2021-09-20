#include <iostream>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "io_flow.h"

#include "io_disp.h"

#include <runningstats/runningstats.h>
using runningstats::RunningCovariance;
using runningstats::QuantileStats;
using runningstats::RunningStats;
using runningstats::BinaryStats;


#define FILENAME "/tmp/BpcpUNzAsu0LEWiAXRboPJW.kitti.png"

#define DISCRETIZATION (1.0/64.0)

testing::AssertionResult IsNegative(double value) {
    if (value < 0) {
        return testing::AssertionSuccess() << value << " < 0";
    }
    else {
        return testing::AssertionFailure() << value << " >= 0";
    }

}

testing::AssertionResult IsEqualFlow(cv::Mat const& mat, KITTI::FlowImage const& img, double const threshold = 0.001) {
    if (mat.rows != img.height()) {
        return testing::AssertionFailure()
                << "Height doesn't match: OpenCV has " << mat.rows << " while KITTI has " << img.height();
    }
    if (mat.cols != img.width()) {
        return testing::AssertionFailure()
                << "Width doesn't match: OpenCV has " << mat.cols << " while KITTI has " << img.width();
    }
    int const cols = mat.cols;
    int const rows = mat.rows;

    for (int v_row_index = 0; v_row_index < rows; v_row_index++) {
        cv::Vec2f const* flow_row = mat.ptr<cv::Vec2f>(v_row_index);
        for (int u_column_index = 0; u_column_index < cols; u_column_index++) {
            float const cv_u = flow_row[u_column_index][0];
            float const cv_v = flow_row[u_column_index][1];
            if (img.isValid(u_column_index, v_row_index)) {
                if (!std::isfinite(cv_u)) {
                    return testing::AssertionFailure() << "KITTI shows valid at (" << u_column_index << ", " << v_row_index << ") while cv_u is not finite";
                }
                if (!std::isfinite(cv_v)) {
                    return testing::AssertionFailure() << "KITTI shows valid at (" << u_column_index << ", " << v_row_index << ") while cv_v is not finite";
                }
                float const kitti_u = img.getFlowU(u_column_index, v_row_index);
                float const kitti_v = img.getFlowV(u_column_index, v_row_index);
                if (std::abs(cv_u - kitti_u) > threshold) {
                    return testing::AssertionFailure() << "Difference in u component at ("
                                                       << u_column_index << ", " << v_row_index << ") between KITTI u ("
                                                       << kitti_u << ") and CV u ("
                                                       << cv_u << ") is "
                                                       << std::abs(cv_u - kitti_u)
                                                       << ", exceeds " << threshold;
                }
                if (std::abs(cv_v - kitti_v) > threshold) {
                    return testing::AssertionFailure() << "Difference in v component at ("
                                                       << u_column_index << ", " << v_row_index << ") between KITTI v ("
                                                       << kitti_v << ") and CV v ("
                                                       << cv_v << ") is "
                                                       << std::abs(cv_v - kitti_v)
                                                       << ", exceeds " << threshold;
                }
            }
            else {
                if (std::isfinite(cv_u) && std::isfinite(cv_v)) {
                    return testing::AssertionFailure() << "KITTI shows invalid at " << u_column_index << ", " << v_row_index << ") while CV u and CV v are finite";
                }
            }
        }
    }


    return testing::AssertionSuccess() << "Images are equal";
}


testing::AssertionResult IsEqualFlow(cv::Mat const& a, cv::Mat const& b, double const threshold = 0.001) {
    if (a.rows != b.rows) {
        return testing::AssertionFailure()
                << "Height doesn't match: Image a has " << a.rows << " while image b has " << b.rows;
    }
    if (a.cols != b.cols) {
        return testing::AssertionFailure()
                << "Width doesn't match: Image a has " << a.cols << " while image b has " << b.cols;
    }
    int const cols = a.cols;
    int const rows = a.rows;

    float max_diff_u = 0;
    float max_diff_v = 0;

    float max_diff_u_up = 0;
    float max_diff_u_down = 0;

    float max_diff_v_up = 0;
    float max_diff_v_down = 0;

    for (int row = 0; row < rows; row++) {
        cv::Vec2f const* a_row = a.ptr<cv::Vec2f>(row);
        cv::Vec2f const* b_row = b.ptr<cv::Vec2f>(row);
        for (int col = 0; col < cols; col++) {
            float const a_u = a_row[col][0];
            float const a_v = a_row[col][1];
            float const b_u = b_row[col][0];
            float const b_v = b_row[col][1];

            if (std::isfinite(a_u) && std::isfinite(a_v)) {
                if (std::isfinite(b_u) && std::isfinite(b_v)) {
                    float const diff_u = a_u - b_u;
                    float const diff_v = a_v - b_v;
                    max_diff_u = std::max(max_diff_u, std::abs(diff_u));
                    max_diff_v = std::max(max_diff_v, std::abs(diff_v));
                    max_diff_u_up = std::max(max_diff_u_up, diff_u);
                    max_diff_v_up = std::max(max_diff_v_up, diff_v);
                    max_diff_u_down = std::min(max_diff_u_down, diff_u);
                    max_diff_v_down = std::min(max_diff_v_down, diff_v);
                    if (std::abs(diff_u) > threshold) {
                        return testing::AssertionFailure()
                                << "Difference in u component at (" << col << ", " << row << ") between image a ("
                                << a_u << ") and image b (" << b_u << ") is " << diff_u
                                << " which exceeds " << threshold << " by "
                                << std::abs(diff_u) - threshold;
                    }
                    if (std::abs(diff_v) > threshold) {
                        return testing::AssertionFailure()
                                << "Difference in v component at (" << col << ", " << row << ") between image a ("
                                << a_v << ") and image b (" << b_v << ") is " << diff_v
                                << " which exceeds " << threshold
                                << " by " << std::abs(diff_v) - threshold;
                    }
                }
                else {
                    return testing::AssertionFailure()
                            << "Image a is valid at (" << col << ", " << row << ") while image b is invalid";
                }
            }
            else {
                if (std::isfinite(b_u) && std::isfinite(b_v)) {
                    return testing::AssertionFailure()
                            << "Image a is invalid at (" << col << ", " << row << ") while image b is valid";
                }
            }

        }
    }
    std::cout << "max diff: " << max_diff_u << ", " << max_diff_v
              << ", ranges: [" << max_diff_u_down << ", "  << max_diff_u_up
              << "], [" << max_diff_v_down << ", " << max_diff_v_up << "]";

    std::cout << " excentricity: " << max_diff_u_down + max_diff_u_up << ", " << max_diff_v_down + max_diff_v_up;

    std::cout << std::endl;


    return testing::AssertionSuccess() << "Images are equal";
}


testing::AssertionResult IsEqualDisp(
        cv::Mat const& mat,
        KITTI::DisparityImage const& img,
        double const threshold = 0.001) {
    if (mat.rows != img.height()) {
        return testing::AssertionFailure()
                << "Height doesn't match: OpenCV has " << mat.rows << " while KITTI has " << img.height();
    }
    if (mat.cols != img.width()) {
        return testing::AssertionFailure()
                << "Width doesn't match: OpenCV has " << mat.cols << " while KITTI has " << img.width();
    }
    int const cols = mat.cols;
    int const rows = mat.rows;

    RunningStats differences;

    for (int v_row_index = 0; v_row_index < rows; v_row_index++) {
        cv::Vec2f const* flow_row = mat.ptr<cv::Vec2f>(v_row_index);
        for (int u_column_index = 0; u_column_index < cols; u_column_index++) {
            float const cv_disp = flow_row[u_column_index][0];
            float const cv_unc = flow_row[u_column_index][1];
            bool const kitti_valid = img.isValid(u_column_index, v_row_index);
            if (kitti_valid) {
                if (!std::isfinite(cv_disp)) {
                    return testing::AssertionFailure() << "KITTI shows valid at (" << u_column_index << ", " << v_row_index << ") while cv_disp is not finite";
                }
                if (!std::isfinite(cv_unc)) {
                    return testing::AssertionFailure() << "KITTI shows valid at (" << u_column_index << ", " << v_row_index << ") while cv_unc is not finite";
                }
                float const kitti_disp = img.getDisp(u_column_index, v_row_index);
                differences.push(cv_disp - kitti_disp);
                if (std::abs(cv_disp - kitti_disp) > threshold) {
                    return testing::AssertionFailure() << "Difference in disparity component at ("
                                                       << u_column_index << ", " << v_row_index << ") between KITTI disp ("
                                                       << kitti_disp << ") and CV disp ("
                                                       << cv_disp << ") is "
                                                       << std::abs(cv_disp - kitti_disp)
                                                       << ", exceeds " << threshold;
                }
                if (std::abs(cv_disp - kitti_disp) > threshold) {
                    return testing::AssertionFailure() << "Difference in v component at ("
                                                       << u_column_index << ", " << v_row_index << ") between KITTI v ("
                                                       << kitti_disp << ") and CV v ("
                                                       << cv_disp << ") is "
                                                       << std::abs(cv_disp - kitti_disp)
                                                       << ", exceeds " << threshold;
                }
            }
            else {
                if (std::isfinite(cv_disp) && std::isfinite(cv_unc)) {
                    return testing::AssertionFailure() << "KITTI shows invalid at " << u_column_index << ", " << v_row_index << ") while CV disp and CV unc are finite";
                }
            }
        }
    }

    std::cout << "Differences: " << differences.print() << std::endl;


    return testing::AssertionSuccess() << "Images are equal";
}


testing::AssertionResult IsEqualDisp(cv::Mat const& a, cv::Mat const& b, double const threshold = 0.001) {
    if (a.rows != b.rows) {
        return testing::AssertionFailure()
                << "Height doesn't match: Image a has " << a.rows << " while image b has " << b.rows;
    }
    if (a.cols != b.cols) {
        return testing::AssertionFailure()
                << "Width doesn't match: Image a has " << a.cols << " while image b has " << b.cols;
    }
    int const cols = a.cols;
    int const rows = a.rows;

    for (int row = 0; row < rows; row++) {
        cv::Vec2f const* a_row = a.ptr<cv::Vec2f>(row);
        cv::Vec2f const* b_row = b.ptr<cv::Vec2f>(row);
        for (int col = 0; col < cols; col++) {
            float const a_disp = a_row[col][0];
            float const a_unc  = a_row[col][1];
            float const b_disp = b_row[col][0];
            float const b_unc  = b_row[col][1];

            if (std::isfinite(a_disp) && std::isfinite(a_unc)) {
                if (std::isfinite(b_disp) && std::isfinite(b_unc)) {
                    if (std::abs(a_disp - b_disp) > threshold) {
                        return testing::AssertionFailure()
                                << "Difference in u component at (" << col << ", " << row << ") between image a ("
                                << a_disp << ") and image b (" << b_disp << ") is " << std::abs(a_disp - b_disp)
                                << " which exceeds " << threshold << " by "
                                << std::abs(a_disp - b_disp) - threshold;
                    }
                }
                else {
                    return testing::AssertionFailure()
                            << "Image a is valid at (" << col << ", " << row << ") while image b is invalid";
                }
            }
            else {
                if (std::isfinite(b_disp) && std::isfinite(b_unc)) {
                    return testing::AssertionFailure()
                            << "Image a is invalid at (" << col << ", " << row << ") while image b is valid";
                }
            }

        }
    }
    return testing::AssertionSuccess() << "Images are equal";
}

TEST(io_flow, conversion1) {
    cv::Mat original (342, 809, CV_32FC2);

    cv::randu(original, -500, 500);

    cv::Mat c_damaged = original.clone();

    KITTI::FlowImage const kitti(original);

    EXPECT_TRUE(IsEqualFlow(original, kitti));

    KITTI::FlowImage k_damaged(kitti);
    k_damaged.setFlowU(0,0, kitti.getFlowU(0,0)+1);
    EXPECT_FALSE(IsEqualFlow(original, k_damaged));

    k_damaged.setFlowU(0,0, kitti.getFlowU(0,0));
    EXPECT_TRUE(IsEqualFlow(original, k_damaged));

    k_damaged.setFlowU(123, 321, kitti.getFlowU(123,321)+1);
    EXPECT_FALSE(IsEqualFlow(original, k_damaged));

    k_damaged.setFlowU(123, 321, kitti.getFlowU(123,321));
    EXPECT_TRUE(IsEqualFlow(original, k_damaged));

    k_damaged.setFlowV(123, 321, kitti.getFlowU(123,321)+1);
    EXPECT_FALSE(IsEqualFlow(original, k_damaged));

    k_damaged.setFlowV(123, 321, kitti.getFlowV(123,321));
    EXPECT_TRUE(IsEqualFlow(original, k_damaged));

    k_damaged.setValid(123, 321, false);
    EXPECT_FALSE(IsEqualFlow(original, k_damaged));

    k_damaged.setValid(123, 321, true);
    EXPECT_TRUE(IsEqualFlow(original, k_damaged));

    {
        c_damaged.at<cv::Vec2f>(0,0)[0] += 1;
        KITTI::FlowImage tmp(c_damaged);
        cv::Mat tmp2 = tmp.getCVMat();
        EXPECT_FALSE(IsEqualFlow(original, tmp2));
    }
    {
        c_damaged.at<cv::Vec2f>(0,0)[0] = original.at<cv::Vec2f>(0,0)[0];
        KITTI::FlowImage tmp(c_damaged);
        cv::Mat tmp2 = tmp.getCVMat();
        EXPECT_TRUE(IsEqualFlow(original, tmp2));
    }
    {
        c_damaged.at<cv::Vec2f>(10,0)[0] = std::numeric_limits<float>::quiet_NaN();
        KITTI::FlowImage tmp(original);
        tmp.setValid(0,10, false);
        cv::Mat tmp2 = tmp.getCVMat();
        EXPECT_TRUE(IsEqualFlow(c_damaged, tmp2));
    }

    {
        c_damaged = original.clone();
        c_damaged.at<cv::Vec2f>(23,42)[0] = std::numeric_limits<float>::quiet_NaN();
        KITTI::FlowImage tmp(c_damaged);
        tmp.write(FILENAME);
        KITTI::FlowImage tmp2;
        tmp2.read(FILENAME);
        cv::Mat read = tmp2.getCVMat();
        EXPECT_TRUE(IsEqualFlow(c_damaged, read, DISCRETIZATION/2));
    }
    {
        c_damaged = original.clone();
        c_damaged.at<cv::Vec2f>(42,23)[1] = std::numeric_limits<float>::quiet_NaN();
        KITTI::FlowImage tmp(c_damaged);
        std::stringstream storage;
        tmp.write(storage);
        KITTI::FlowImage tmp2;
        tmp2.read(storage);
        cv::Mat read = tmp2.getCVMat();
        EXPECT_TRUE(IsEqualFlow(c_damaged, read, DISCRETIZATION/2));
    }

}

TEST(io_disp, conversion1) {
    cv::Mat original (342, 809, CV_32FC2);

    cv::randu(original, 0, 255);
    original.at<cv::Vec2f>(0,0)[0] = 0;


    cv::Mat tmp;
    {
        KITTI::DisparityImage const kitti(original);
        EXPECT_TRUE(IsEqualDisp(original, kitti, DISCRETIZATION));
        tmp = kitti.getCVMat();
        EXPECT_TRUE(IsEqualDisp(original, tmp, DISCRETIZATION));

        kitti.write(FILENAME);
        KITTI::DisparityImage const recovered(FILENAME);
        EXPECT_TRUE(IsEqualDisp(original, recovered, DISCRETIZATION));
        tmp = recovered.getCVMat();
        EXPECT_TRUE(IsEqualDisp(original, tmp, DISCRETIZATION));
    }
    {
        cv::Mat c_damaged = original.clone();
        c_damaged.at<cv::Vec2f>(23,42)[0] += 10;
        KITTI::DisparityImage const kitti(original);
        EXPECT_FALSE(IsEqualDisp(c_damaged, kitti, DISCRETIZATION));
        tmp = kitti.getCVMat();
        EXPECT_FALSE(IsEqualDisp(c_damaged, tmp, DISCRETIZATION));

    }
    {
        cv::Mat c_damaged = original.clone();
        c_damaged.at<cv::Vec2f>(23,42)[0] = std::numeric_limits<float>::quiet_NaN();
        KITTI::DisparityImage kitti(original);
        EXPECT_FALSE(IsEqualDisp(c_damaged, kitti, DISCRETIZATION));
        tmp = kitti.getCVMat();
        EXPECT_FALSE(IsEqualDisp(c_damaged, tmp, DISCRETIZATION));

        kitti.setInvalid(42,23);
        EXPECT_TRUE(IsEqualDisp(c_damaged, kitti, DISCRETIZATION));
        tmp = kitti.getCVMat();
        EXPECT_TRUE(IsEqualDisp(c_damaged, tmp, DISCRETIZATION));

        kitti.write(FILENAME);
        KITTI::DisparityImage const recovered(FILENAME);
        EXPECT_TRUE(IsEqualDisp(c_damaged, recovered, DISCRETIZATION));
        tmp = kitti.getCVMat();
        EXPECT_TRUE(IsEqualDisp(c_damaged, tmp, DISCRETIZATION));
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;
}
