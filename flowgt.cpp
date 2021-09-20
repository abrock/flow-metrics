
#include "readgt.h"

#include "flowgt.h"



#include "stringexception.h"

#include "basename.h"
#include <boost/filesystem.hpp>
#include <stdlib.h>
#include <chrono>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"

#include <opencv2/imgproc/types_c.h>
#include <opencv2/optflow.hpp>
#include <opencv2/core.hpp>

double GT::percent(double a, double b) {
    return 100.0 * a/b;
}

/**
 * @brief percent calculates percentages
 * @param a
 * @param mat
 * @return
 */
double GT::percent(double a, cv::Mat const& mat) {
    return percent(a, mat.total());
}

static cv::Mat linspace(float x0, float x1, int n)
{
    cv::Mat pts(n, 1, CV_32FC1);
    float step = (x1-x0)/(n-1);
    for(int i = 0; i < n; i++)
        pts.at<float>(i,0) = x0+i*step;
    return pts;
}


void GT::adjust_size(cv::Mat const& gt, cv::Mat & submission) {
    if (gt.size == submission.size) {
        return;
    }

    if (submission.cols > gt.cols || submission.rows > gt.rows) {
        cv::Rect roi(0, 0, std::min(gt.cols, submission.cols), std::min(gt.rows, submission.rows));
        submission = submission(roi).clone();
    }

    if (
            (gt.rows > submission.rows && gt.cols >= submission.cols)
            || (gt.cols > submission.cols && gt.rows >= submission.rows)
            ) {
        cv::copyMakeBorder(
                    submission, submission,
                    0, gt.rows - submission.rows,
                    0, gt.cols - submission.cols,
                    cv::BORDER_REPLICATE);
    }
}

std::string GT::matsize(cv::Mat const& mat) {
    return std::string("(") + std::to_string(mat.cols) + "x" + std::to_string(mat.rows) + ")";
}

double GT::getSparsityVisu(cv::Mat const& mat, cv::Mat & visu) {
    size_t invalid = 0;
    int const cols = mat.cols;
    int const rows = mat.rows;
    visu = cv::Mat(rows, cols, CV_8UC3);
    if (0 == cols  || 0 == rows) {
        return 1.0;
    }
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const mat_row = mat.ptr<cv::Vec2f>(ii);
        cv::Vec3b * visu_row = visu.ptr<cv::Vec3b>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            visu_row[jj] = cv::Vec3b(0,0,0);
            if (!isValidFlow(mat_row[jj])) {
                invalid++;
                visu_row[jj] = cv::Vec3b(0,0,255);
            }
        }
    }
    return (double)invalid / (cols * rows);
}

double GT::dist_sq(const cv::Vec2f a, const cv::Vec2f b) {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
}

double GT::dist(const cv::Vec2f a, const cv::Vec2f b) {
    return std::sqrt(dist_sq(a, b));
}

cv::Mat GT::readUncertainty(const fs::path &filename) {
    cv::Mat result;
    if (filename.extension() == std::string(".png")) {
        result = cv::imread(filename.string(), cv::IMREAD_UNCHANGED);
        result.convertTo(result, CV_32FC1);
        result /= 255;
        return result;
    }
    else {
        throw std::runtime_error("File format for flow uncertainty not supported");
    }
    return result;
}

#include "kitti-devkit/io_flow.h"

cv::Mat GT::readOpticalFlow(std::string const& filename) {
    if (".flo" == filename.substr(filename.length() - 4)) {
        return cv::readOpticalFlow(filename);
    }
    if (".png" == filename.substr(filename.length() - 4)) {
        return KITTI::FlowImage(filename).getCVMat();
    }

    throw std::runtime_error("Method not implemented for given file");
}

cv::Mat GT::readOpticalFlow(fs::path const& filename) {
    return readOpticalFlow(filename.string());
}

cv::Mat GT::readOpticalFlow(const char* filename) {
    return readOpticalFlow(std::string(filename));
}

std::string GT::printStats(const cv::Mat &flow) {
    int rows = flow.rows;
    int cols = flow.cols;
    size_t counter = 0;
    size_t valid_counter = 0;
    RunningStats stat_u, stat_v, stat_u_abs, stat_v_abs, stat_length;
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _flow = flow.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            cv::Vec2f const current = _flow[jj];
            counter++;
            if (isValidFlow(current)) {
                valid_counter++;
                stat_u.push(current[0]);
                stat_u_abs.push(std::abs(current[0]));
                stat_v.push(current[1]);
                stat_v_abs.push(std::abs(current[1]));
                stat_length.push(std::sqrt(current[0]*current[0] + current[1]*current[1]));
            }
        }
    }
    std::stringstream out;
    out << "u:      " << stat_u.print() << std::endl
        << "abs(u): " << stat_u_abs.printBoth() << std::endl
        << "v:      " << stat_v.print() << std::endl
        << "abs(v): " << stat_v_abs.printBoth() << std::endl
        << "length: " << stat_length.printBoth() << std::endl
        << "valid:  " << 100.0 * double(valid_counter) / double(counter) << "%" << std::endl;
    return out.str();
}

cv::Mat_<cv::Vec3b> GT::colorFlow(
        const cv::Mat_<cv::Vec2f>& flow,
        const float factor,
        const double scaleFactor) {
    makecolorwheel();
    cv::Mat_<cv::Vec3b> result(flow.rows, flow.cols, cv::Vec3b(0,0,0));
#pragma omp parallel for
    for (int ii = 0; ii < flow.rows; ++ii) {
        cv::Vec3b * resultRow = result.ptr<cv::Vec3b>(ii);
        const cv::Vec2f * flowRow = flow.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < flow.cols; ++jj) {
            computeColor(
                        flowRow[jj][0]/factor,
                    flowRow[jj][1]/factor,
                    resultRow[jj]);
        }
    }
    if (scaleFactor > 0) {
        cv::resize(result, result, cv::Size(0,0), scaleFactor, scaleFactor, cv::INTER_CUBIC);
    }
    return result;
}

cv::Mat GT::grayFlow(const cv::Mat_<cv::Vec2f> &flow, double threshold)
{
    int const cols = flow.cols;
    int const rows = flow.rows;

    cv::Mat result(rows, cols, CV_8UC3, cv::Vec3b(0,0,0));

    if (threshold <= 0) {
        threshold = maxFlowLength(flow);
    }
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const ptr = flow.ptr<cv::Vec2f>(ii);
        cv::Vec3b * dst = result.ptr<cv::Vec3b>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            const double length = std::sqrt(ptr[jj][0] * ptr[jj][0] + ptr[jj][1] * ptr[jj][1]);
            const double normalized = length * 255.0 / threshold;
            const uint8_t len = (uint8_t)std::min(std::round(normalized), 255.0);
            dst[jj] = cv::Vec3b(len, len, len);
        }
    }

    return result;
}

void GT::flowStats(cv::Mat const& flow, runningstats::RunningCovariance *covar, runningstats::RunningStats *length, cv::Mat const& mask) {
    bool const has_mask = (flow.size == mask.size);
    float min_u, min_v, max_u, max_v;
    min_u = min_v = std::numeric_limits<float>::max();
    max_u = max_v = std::numeric_limits<float>::lowest();
    int const rows = flow.rows;
    int const cols = flow.cols;
    if (flow.type() != CV_32FC2) {
        throw std::runtime_error("Flow mat has not type CV_32FC2");
    }
    if (has_mask && mask.type() != CV_8UC1) {
        throw std::runtime_error("Mask mat has not type CV_8UC1");
    }
    double maxlength = 0;
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const ptr = flow.ptr<cv::Vec2f>(ii);
        uint8_t const * const ptr_mask = has_mask ? mask.ptr<uint8_t>(ii) : nullptr;
        for (int jj = 0; jj < cols; ++jj) {
            if (GT::isValidFlow(ptr[jj])) {
                if (!has_mask || ptr_mask[jj] > 0) {
                    if (length != nullptr) {
                        length->push(std::sqrt(ptr[jj][0] * ptr[jj][0] + ptr[jj][1] * ptr[jj][1]));
                    }
                    if (covar != nullptr) {
                        covar->push(ptr[jj][0], ptr[jj][1]);
                    }
                }
            }
        }
    }
}

double GT::maxFlowLength(const cv::Mat &flow,
                         float *_min_u,
                         float *_max_u,
                         float *_min_v,
                         float *_max_v)
{
    float min_u, min_v, max_u, max_v;
    min_u = min_v = std::numeric_limits<float>::max();
    max_u = max_v = std::numeric_limits<float>::lowest();
    int const rows = flow.rows;
    int const cols = flow.cols;
    if (flow.type() != CV_32FC2) {
        throw std::runtime_error("Matrix has not type CV_32FC2");
    }
    double maxlength_squared = 0;
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const ptr = flow.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            if (GT::isValidFlow(ptr[jj])) {
                const double length_squared = ptr[jj][0] * ptr[jj][0] + ptr[jj][1] * ptr[jj][1];
                if (length_squared > maxlength_squared) {
                    maxlength_squared = length_squared;
                }
                min_u = std::min(min_u, ptr[jj][0]);
                max_u = std::max(max_u, ptr[jj][0]);
                min_v = std::min(min_v, ptr[jj][1]);
                max_v = std::max(max_v, ptr[jj][1]);
            }
        }
    }
    if (_min_u != nullptr) {
        *_min_u = min_u;
    }
    if (_min_v != nullptr) {
        *_min_v = min_v;
    }
    if (_max_u != nullptr) {
        *_max_u = max_u;
    }
    if (_max_v != nullptr) {
        *_max_v = max_v;
    }
    return std::sqrt(maxlength_squared);
}

double GT::flowLength(cv::Vec2f const vec) {
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
}

void GT::colorFlow(const cv::Mat& flow, const std::string& filename, const double factor, const double scaleFactor) {
    cv::Mat result = colorFlow(flow, factor, scaleFactor);
    cv::imwrite(filename + ".png", result);
}

bool GT::isValidFlow(cv::Vec2f const& vec) {
    return
            std::isfinite(vec[0]) && std::isfinite(vec[1])
            && vec[0] <= +1e8 && vec[1] <= +1e8
            && vec[0] >= -1e8 && vec[1] >= -1e8;
}

cv::Vec2f GT::invalidFlow()
{
    return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
}

cv::Scalar GT::invalidFlowScalar()
{
    return cv::Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
}

template<class Doc, class String1, class Vector, class Object>
void GT::addStats(Doc& doc, String1 key1, QuantileStats<float>& stats, Vector& quantiles, Object& object) {
    rapidjson::Value innerObject;

    addDouble(doc, "mean", stats.getMean(), innerObject);
    addDouble(doc, "stddev", stats.getStddev(), innerObject);

    for (auto quantile : quantiles) {
        addDouble(doc, "quantile " + std::to_string(quantile), stats.getQuantile(quantile), innerObject);
    }

    addMember(doc, key1, innerObject, object);

}

template<class Doc, class String1, class String2, class String3, class Object>
void GT::addDoubleString(Doc& doc, String1 key1, String2 key2, String3 value, Object& object) {
    rapidjson::Value innerObject;
    addString(doc, key2, value, innerObject);
    addMember(doc, key1, innerObject, object);

}

template<class Doc, class String, class Object>
void GT::addScore(Doc& doc, String scoreName, double scoreValue, Object& scores) {
    rapidjson::Value container;
    container.SetObject();
    rapidjson::Value value;
    value.SetDouble(scoreValue);
    addMember(doc, "value", value, container);
    addMember(doc, scoreName, container, scores);
}

template<class Doc, class String, class Object>
void GT::addDouble(Doc& doc, String name, double doubleValue, Object& scores) {
    rapidjson::Value value;
    value.SetDouble(doubleValue);
    addMember(doc, name, value, scores);
}


template<class Doc, class String1, class String2, class Object>
void GT::addString(Doc& doc, String1 key, String2 value, Object& object) {
    rapidjson::Value val;
    val.SetString(std::string(value).c_str(), std::string(value).size(), doc.GetAllocator());
    addMember(doc, key, val, object);
}

template<class Doc, class Member, class String, class Object>
/**
 * @brief addMember Adds a member to an object in a rapidjson::Document
 * @param doc Document which is needed for getting the allocator.
 * @param name Name of the new member added.
 * @param member Member which shall be added.
 * @param object Object within the document to which the member should be added.
 * May also be the document itself.
 */
void GT::addMember(Doc& doc, String name, Member& member, Object& object) {
    if (!doc.IsObject()) {
        doc.SetObject();
    }
    if (!object.IsObject()) {
        object.SetObject();
    }

    object.AddMember(rapidjson::Value(std::string(name).c_str(), doc.GetAllocator()).Move(), member, doc.GetAllocator());
}

void GT::badpix(cv::Mat const& gt,
                cv::Mat const& submission,
                cv::Mat const& unc,
                cv::Mat const& valid_mask,
                cv::Mat const& fattening_mask,
                RunningStats & result,
                cv::Mat& visu,
                RunningStats &result_fat,
                cv::Mat& visu_fat,
                double const threshold) {

    int const cols = gt.cols;
    int const rows = gt.rows;

    bool const has_unc   = (gt.size == unc.size);
    bool const has_valid = (gt.size == valid_mask.size);
    bool const has_fat   = (gt.size == fattening_mask.size);

    visu = GT::grayFlow(submission, GT::maxFlowLength(gt));
    if (has_fat) {
        visu_fat = GT::grayFlow(submission, GT::maxFlowLength(gt));
    }

    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const gt_row = gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const sub_row = submission.ptr<cv::Vec2f>(ii);
        float const * const unc_row = has_unc ? unc.ptr<float>(ii) : nullptr;
        uint8_t const * const fat_row = has_fat ? fattening_mask.ptr<uint8_t>(ii) : nullptr;
        uint8_t const * const valid_row = has_fat ? valid_mask.ptr<uint8_t>(ii) : nullptr;
        cv::Vec3b * visu_row = visu.ptr<cv::Vec3b>(ii);
        cv::Vec3b * visu_fat_row = has_fat ? visu_fat.ptr<cv::Vec3b>(ii) : nullptr;
        for (int jj = 0; jj < cols; ++jj) {
            if (isValidFlow(gt_row[jj])) {
                cv::Vec2f const residual = gt_row[jj] - sub_row[jj];
                float const err = std::sqrt(residual.dot(residual));
                if (!has_valid || (valid_row[jj] == 0)) {
                    if ((!has_unc || (unc_row[jj] < threshold)) && err > threshold) {
                        visu_row[jj] = cv::Vec3b(0,0,255);
                        result.push(1);
                    }
                    else {
                        result.push(0);
                    }
                }
                if (has_fat && fat_row[jj] != 0) {
                    if (!has_valid || (valid_row[jj] == 0)) {
                        if ((!has_unc || (unc_row[jj] < threshold)) && err > threshold) {
                            visu_fat_row[jj] = cv::Vec3b(0,0,255);
                            result_fat.push(1);
                        }
                        else {
                            result_fat.push(0);
                            visu_fat_row[jj] = cv::Vec3b(0,255,0);
                        }
                    }
                }
            }
            if (has_valid && valid_row[jj] != 0) {
                visu_row[jj] = invalid_color;
                if (has_fat && fat_row[jj] != 0) {
                    visu_fat_row[jj] = invalid_color;
                }
            }
            if (has_unc && unc_row[jj] > threshold) {
                visu_row[jj] = invalid_color;
                if (has_fat && fat_row[jj] != 0) {
                    visu_fat_row[jj] = invalid_color;
                }
            }
        }
    }
}

double maha_normalization_factor(
        cv::Mat const& unc,
        cv::Mat const& mask
        ) {
    int const rows = unc.rows;
    int const cols = unc.cols;
    bool const has_mask = (unc.size == mask.size);

    double sum = 0;
    size_t counter = 0;
    for (int ii = 0; ii < rows; ++ii) {
        float const * const _unc = unc.ptr<float>(ii);
        float const * const _mask = has_mask ? mask.ptr<float>(ii) : nullptr;
        for (int jj = 0; jj < cols; ++jj) {
            if (!has_mask || _mask[jj] == 0) {
                sum += _unc[jj];
                counter++;
            }
        }
    }
    if (counter == 0) {
        return 1;
    }
    return sum / counter;
}

void GT::evaluate(
        std::string const& visualization_prefix,
        cv::Mat const &gt,
        cv::Mat const &submission,
        cv::Mat const &unc,
        cv::Mat const &valid_mask,
        cv::Mat const &fattening_mask,
        const boost::filesystem::path &dst_dir,
        json::JSON &json_result,
        json::JSON &metrics_definitions) {

    int const cols = gt.cols;
    int const rows = gt.rows;
    cv::Mat visu(rows, cols, CV_8UC3, cv::Vec3b(0,0,0));

    bool const has_unc   = (gt.size == unc.size);
    bool const has_valid = (gt.size == valid_mask.size);
    bool const has_fat   = (gt.size == fattening_mask.size);

    double const maha_factor = maha_normalization_factor(unc, valid_mask);

    cv::Mat visu_fat;
    if (has_fat) {
        visu_fat = GT::grayFlow(submission, GT::maxFlowLength(gt));
    }

    cv::Mat visu_maha;
    if (has_unc) {
        visu_maha = cv::Mat(rows, cols, CV_8UC3, cv::Vec3b(0,0,0));
    }

    RunningStats MEE, Maha, MEE_fat, Maha_fat;
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const gt_row = gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const sub_row = submission.ptr<cv::Vec2f>(ii);
        float const * const unc_row = has_unc ? unc.ptr<float>(ii) : nullptr;
        uint8_t const * const fat_row = has_fat ? fattening_mask.ptr<uint8_t>(ii) : nullptr;
        uint8_t const * const valid_row = has_fat ? valid_mask.ptr<uint8_t>(ii) : nullptr;
        cv::Vec3b * visu_row = visu.ptr<cv::Vec3b>(ii);
        cv::Vec3b * visu_fat_row = has_fat ? visu_fat.ptr<cv::Vec3b>(ii) : nullptr;
        cv::Vec3b * visu_maha_row = has_unc ? visu_maha.ptr<cv::Vec3b>(ii) : nullptr;
        for (int jj = 0; jj < cols; ++jj) {
            if (isValidFlow(gt_row[jj]) && (!has_valid || valid_row[jj] == 0)) {
                cv::Vec2f const residual = gt_row[jj] - sub_row[jj];
                float const err = std::sqrt(residual.dot(residual));
                MEE.push(err);
                uint8_t err_visu = cv::saturate_cast<uint8_t>(err*255.0/3.0);
                visu_row[jj] = cv::Vec3b(err_visu, err_visu, err_visu);
                if (has_unc && unc_row[jj] > 1e-6) {
                    float const local_unc = unc_row[jj];
                    cv::Vec2f const maha_res = residual/local_unc;
                    float const maha_err = std::sqrt(maha_res.dot(maha_res));
                    Maha.push(maha_err);
                    uint8_t maha_err_visu = cv::saturate_cast<uint8_t>(maha_err*255.0/3.0*maha_factor);
                    visu_maha_row[jj] = cv::Vec3b(maha_err_visu, maha_err_visu, maha_err_visu);
                }
                if (has_fat && fat_row[jj] != 0) {

                }
            }
            if ((has_valid && valid_row[jj] != 0) || !isValidFlow(gt_row[jj])) {
                visu_row[jj] = invalid_color;
                if (has_unc) {
                    visu_maha_row[jj] = invalid_color;
                }
                if (has_fat) {

                }
            }
        }
    }
    std::string MEE_name = "mee";
    std::string mee_filename = visualization_prefix + "-" + MEE_name + ".png";
    std::string mee_thumb_filename = visualization_prefix + "-" + MEE_name + "-thumb.png";
    cv::imwrite((dst_dir / mee_filename).string(), visu);
    cv::resize(visu, visu, cv::Size(), .2, .2, cv::InterpolationFlags::INTER_LANCZOS4);
    cv::imwrite((dst_dir / mee_thumb_filename).string(), visu);
    json_result[visualization_prefix]["scores"][MEE_name]["value"] = MEE.getMean();
    json_result[visualization_prefix]["scores"][MEE_name]["visualization"]["thumb"] = mee_thumb_filename;
    json_result[visualization_prefix]["scores"][MEE_name]["visualization"]["large"] = mee_filename;

    metrics_definitions["metrics"][MEE_name] = {"category", "General",
                                                "description", "Mean endpoint error.",
                                                "display_name", "MEE"};
    metrics_definitions["metrics"][MEE_name]["result_table_visualization"] = {
            "is_visible", true,
            "legend", "black = no error, white = high error, blue = no evaluation"};

    if (has_unc) {
        std::string Maha_name = "maha";
        std::string maha_filename = visualization_prefix + "-" + Maha_name + ".png";
        std::string maha_thumb_filename = visualization_prefix + "-" + Maha_name + "-thumb.png";
        cv::imwrite((dst_dir / maha_filename).string(), visu_maha);
        cv::resize(visu_maha, visu_maha, cv::Size(), .2, .2, cv::InterpolationFlags::INTER_LANCZOS4);
        cv::imwrite((dst_dir / maha_thumb_filename).string(), visu_maha);
        json_result[visualization_prefix]["scores"][Maha_name]["value"] = Maha.getMean();
        json_result[visualization_prefix]["scores"][Maha_name]["visualization"]["thumb"] = maha_thumb_filename;
        json_result[visualization_prefix]["scores"][Maha_name]["visualization"]["large"] = maha_filename;

        metrics_definitions["metrics"][Maha_name] = {"category", "General",
                                                     "description", "Mahalanobis endpoint error, i.e. mean endpoint error weighted with the uncertainties of the ground truth.",
                                                     "display_name", "MEE"};
        metrics_definitions["metrics"][Maha_name]["result_table_visualization"] = {
                "is_visible", true,
                "legend", "black = no error, white = high error, blue = no evaluation"};
    }


    std::vector<float> badpix_thresholds = {1,3,10};
    for (float const threshold : badpix_thresholds) {
        cv::Mat bp_visu, bp_visu_fat;
        RunningStats result, result_fat;
        badpix(gt, submission, unc, valid_mask, fattening_mask, result, bp_visu, result_fat, bp_visu_fat, threshold);
        std::stringstream _threshold_name;
        _threshold_name << threshold;
        std::string threshold_name = _threshold_name.str();
        std::string bp_name = std::string("bad_pix_") + threshold_name;
        std::string bp_filename = visualization_prefix + "-" + bp_name + ".png";
        json_result[visualization_prefix]["scores"][bp_name]["value"] = result.getMean() * 100;
        json_result[visualization_prefix]["scores"][bp_name]["visualization"]["large"] = bp_filename;
        cv::imwrite((dst_dir / bp_filename).string(), bp_visu);

        std::string bp_thumb_filename = visualization_prefix + "-" + bp_name + "-thumb.png";
        cv::resize(bp_visu, bp_visu, cv::Size(), .2, .2, cv::InterpolationFlags::INTER_LANCZOS4);
        json_result[visualization_prefix]["scores"][bp_name]["visualization"]["thumb"] = bp_thumb_filename;
        cv::imwrite((dst_dir / bp_thumb_filename).string(), bp_visu);

        metrics_definitions["metrics"][bp_name] = {"category", "General",
                                                   "description", std::string("Percentage of pixels where the endpoint error is larger than ") + threshold_name + "px.",
                                                   "display_name", bp_name};
        metrics_definitions["metrics"][bp_name]["result_table_visualization"] = {
                "is_visible", true,
                "legend", "red = error above threshold, gray = flow length of result"};

        if (has_fat) {
            std::string bp_name = std::string("bad_pix_") + threshold_name + "_discont";
            std::string bp_filename = visualization_prefix + "-" + bp_name + ".png";
            json_result[visualization_prefix]["scores"][bp_name]["value"] = result_fat.getMean() * 100;
            json_result[visualization_prefix]["scores"][bp_name]["visualization"]["large"] = bp_filename;
            cv::imwrite((dst_dir / bp_filename).string(), bp_visu_fat);

            std::string bp_thumb_filename = visualization_prefix + "-" + bp_name + "-thumb.png";
            cv::resize(bp_visu_fat, bp_visu_fat, cv::Size(), .2, .2, cv::InterpolationFlags::INTER_LANCZOS4);
            json_result[visualization_prefix]["scores"][bp_name]["visualization"]["thumb"] = bp_thumb_filename;
            cv::imwrite((dst_dir / bp_thumb_filename).string(), bp_visu_fat);

            metrics_definitions["metrics"][bp_name] = {"category", "Discontinuity",
                                                       "description", std::string("Percentage of pixels near motion discontinuities where the endpoint error is larger than ") + threshold_name + "px.",
                                                       "display_name", bp_name};
            metrics_definitions["metrics"][bp_name]["result_table_visualization"] = {
                    "is_visible", true,
                    "legend", "red = error above threshold, green = error below threshold, gray = flow length of result"};
        }
    }
}

#include "contourfeatures.h"
typedef std::map<std::string, std::vector<ContourFeatures > > Map1;
typedef rapidjson::Document Doc1;
typedef std::string String1;
typedef std::vector<float> Vec1;
typedef rapidjson::Value Obj1;

template void GT::addStats(Doc1& doc, String1 key1, QuantileStats<float>& stats, Vec1& quantiles, Obj1& object);
template void GT::addStats(Doc1& doc, String1 key1, QuantileStats<float>& stats, Vec1& quantiles, Doc1& object);

template void GT::addString(Doc1&, const char* , const char*, Doc1&);

template void GT::addString(Doc1&, const char*, std::string, rapidjson::Value&);
template void GT::addDoubleString(Doc1&, const char*, const char*, std::string, rapidjson::Value&);
template void GT::addMember(Doc1&, const char*, rapidjson::Value&, Doc1&);



// colorcode.cpp
//
// Color encoding of flow vectors
// adapted from the color circle idea described at
//   http://members.shaw.ca/quadibloc/other/colint.htm
//
// Daniel Scharstein, 4/2007
// added tick marks and out-of-range coding 6/05/07
int GT::colorwheel[MAXCOLS][3];
int GT::ncols = 0;

void GT::setcols(int r, int g, int b, int k) {
colorwheel[k][0] = r;
colorwheel[k][1] = g;
colorwheel[k][2] = b;
}

cv::Size GT::convertSize(cv::MatSize const& src) {
return cv::Size(src[1], src[0]);
}



void GT::computeColor(float fx, float fy, cv::Vec3b &pix) {
if (!isValidFlow(cv::Vec2f(fx, fy))) {
pix[0] = pix[1] = pix[2] = 0;
return;
}

float rad = sqrt(fx * fx + fy * fy);
float a = atan2(-fy, -fx) / M_PI;
float fk = (a + 1.0) / 2.0 * (ncols-1);
int k0 = (int)fk;
int k1 = (k0 + 1) % ncols;
float f = fk - k0;
//f = 0; // uncomment to see original color wheel
for (int b = 0; b < 3; b++) {
                    float col0 = colorwheel[k0][b] / 255.0;
                    float col1 = colorwheel[k1][b] / 255.0;
                    float col = (1 - f) * col0 + f * col1;
                    if (rad <= 1) {
                    col = 1 - rad * (1 - col); // increase saturation with radius
                    }
                    else {
                    col *= .75; // out of range
                    }
                    pix[2 - b] = (int)(255.0 * col);
                    }
                    }



                    GT::GT() {
                    makecolorwheel();
                    }


                    bool GT::hasColorWheel = false;

                    void GT::makecolorwheel() {
                    if (hasColorWheel) {
                    return;
                    }
                    hasColorWheel = true;
                    // relative lengths of color transitions:
                    // these are chosen based on perceptual similarity
                    // (e.g. one can distinguish more shades between red and yellow
                    //  than between yellow and green)
                    int RY = 15;
                    int YG = 6;
                    int GC = 4;
                    int CB = 11;
                    int BM = 13;
                    int MR = 6;
                    ncols = RY + YG + GC + CB + BM + MR;
                    //printf("ncols = %d\n", ncols);
                    if (ncols > MAXCOLS) {
    exit(1);
}
int i;
int k = 0;
for (i = 0; i < RY; i++) {
    setcols(255, 255*i/RY, 0, k++);
}
for (i = 0; i < YG; i++) {
    setcols(255-255*i/YG, 255,		 0,	       k++);
}
for (i = 0; i < GC; i++) {
    setcols(0,		   255,		 255*i/GC,     k++);
}
for (i = 0; i < CB; i++) {
    setcols(0,		   255-255*i/CB, 255,	       k++);
}
for (i = 0; i < BM; i++) {
    setcols(255*i/BM,	   0,		 255,	       k++);
}
for (i = 0; i < MR; i++) {
    setcols(255,	   0,		 255-255*i/MR, k++);
}
}
