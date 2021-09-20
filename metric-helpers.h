#ifndef METRICHELPERS_H
#define METRICHELPERS_H

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/optflow.hpp>
#include "flowgt.h"
#include <stack>
#include <vector>
#include <map>
#include <ParallelTime/paralleltime.h>
#include <simplejson/simplejson.hpp>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::string runEvaluationSingleGtMultipleAlgos(
        std::string const& gtFile,
        const std::string &uncFile,
        std::vector<std::string> const& submissionFiles,
        bool const save_all_images = false);

/**
 * @brief fillFlowHoles fills holes of invalid pixels (NaN, inf, >1e8, <-1e8) in an optical flow field (cv::Vec2f, CV_32FC2)
 * by using the arithmetic mean of all valid pixels in an 8-neighbourhood.
 * This process is repeated until all pixels are valid.
 * If all pixels are invalid the algorithm returns a zero flow field.
 * @param _src
 * @return
 */
cv::Mat_<cv::Vec2f> fillFlowHoles(cv::Mat const& _src);

cv::Vec2f abs(cv::Vec2f const src);

cv::Vec2f max(cv::Vec2f const a, cv::Vec2f const b);

cv::Vec2f absmax(cv::Vec2f const& a, cv::Vec2f const& b);

cv::Mat fill_unc(cv::Mat const& _src);

void ScharrVec2f(
        cv::Mat const& src,
        cv::Mat & dst_x,
        cv::Mat & dst_y,
        int const scale = 1,
        int const delta = 0,
        int const border = cv::BORDER_DEFAULT);

cv::Mat flowGradientMagnitudes(cv::Mat const& src);

void CallBackFunc(int event, int x, int y, int flags, void* userdata);



void CallBackFunc(int event, int x, int y, int flags, void* userdata);


std::vector<fs::path> crawl_dir_rec(const boost::filesystem::path &path);
std::vector<fs::path> crawl_dir_rec(const std::string &path);

bool same_filename(fs::path const& a, fs::path const& b);

void evaluateMetrics(std::string const gt_id,
        cv::Mat const& gt,
        cv::Mat const& submission,
        cv::Mat const& unc,
        cv::Mat const& valid_mask,
        cv::Mat const& fattening_mask,
        fs::path const& dst_dir,
        json::JSON& json_result,
        json::JSON& metrics_definitions,
        const bool all_metrics = false);

bool filenames_match(
        fs::path const& gt_dir,
        fs::path const& gt_name,
        fs::path const& submission_dir,
        fs::path const& submission);

bool filenames_match(
        fs::path const& gt_name,
        fs::path const& submission);

std::map<fs::path, fs::path> matchFiles(
        fs::path const& src_dir,
        std::vector<fs::path> const& src,
        fs::path const& dst_dir,
        std::vector<fs::path> const& dst
        );

template<class A, class B>
bool is_inverse_function(std::map<A, B> const& forward, std::map<B, A> const& reverse);

std::string ltrim_dot(std::string const& in);
std::vector<fs::path> filter_by_ext(std::vector<fs::path> const& input, std::vector<std::string> const& valid_extensions);

#endif // METRICHELPERS_H
