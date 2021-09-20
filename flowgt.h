#ifndef FLOWGT_H
#define FLOWGT_H

#include <opencv2/core.hpp>
#include <runningstats/runningstats.h>
#include <boost/filesystem.hpp>
#include <simplejson/simplejson.hpp>

#include <rapidjson/document.h>

using runningstats::QuantileStats;
using runningstats::RunningStats;

namespace fs=boost::filesystem;

class GT {
private:
    static int ncols;
    #define MAXCOLS 60
    static int colorwheel[MAXCOLS][3];
    static bool hasColorWheel;


    static void setcols(int r, int g, int b, int k);

    static void makecolorwheel();

    static void computeColor(float fx, float fy, cv::Vec3b &pix);


public:

    /**
     * @brief flowStats Compute statistics for flow length as well as u and v components.
     * @param flow Optical flow matrix.
     * @param covar Container for statistics on u and v and their covariance/correlation.
     * @param length Container for statistics on flow lengths.
     * @param mask Optional mask, statistics are only computed where the mask value is positive.
     */
    static void flowStats(
            cv::Mat const& flow,
            runningstats::RunningCovariance * covar = nullptr,
            runningstats::RunningStats * length = nullptr,
            cv::Mat const& mask = cv::Mat());

    /**
     * @brief percent calculates percentages (e.g. for a=1, b=2 the result is 50, for a=1 b=4 the result is 25)
     * @param a The amount relative to b
     * @param b The total amount of stuff, this is 100% by definition
     * @return 100 * a/b
     */
    static double percent(double a, double b);

    /**
     * @brief percent calculates percentages
     * @param a
     * @param mat
     * @return
     */
    static double percent(double a, cv::Mat const& mat);

    static void adjust_size(cv::Mat const& gt, cv::Mat &submission);

    static cv::Scalar invalidFlowScalar();

    static double getSparsity(cv::Mat const& mat);


    static double getSparsityVisu(cv::Mat const& mat, cv::Mat &visu);

    static std::string matsize(cv::Mat const& mat);

    static cv::Mat readUncertainty(fs::path const& filename);

    void evaluate(
            std::string const& visualization_prefix,
            cv::Mat const &gt,
            cv::Mat const &submission,
            cv::Mat const &unc,
            cv::Mat const &valid_mask,
            cv::Mat const &fattening_mask,
            const boost::filesystem::path &dst_dir,
            json::JSON &json_result,
            json::JSON &metrics_definitions);

    static double dist_sq(cv::Vec2f const a, cv::Vec2f const b);
    static double dist(cv::Vec2f const a, cv::Vec2f const b);

    static bool isValidFlow(cv::Vec2f const& vec);

    static cv::Vec2f invalidFlow();

    GT();

    static cv::Mat readOpticalFlow(std::string const& filename);
    static cv::Mat readOpticalFlow(fs::path const& filename);
    static cv::Mat readOpticalFlow(const char* filename);

    const double outputScaling = .25;

    const int FLOW_VALID_COUNT_THRESHOLD = 50;
    double sqr(double val) {
        return val*val;
    }

    static std::string printStats(cv::Mat const& flow);

    /**
     * Evil hack since the uncertainty in the ground truth is wrong atm.
     * This should be set to 1.0 or removed.
     */
    double uncScale = 1;

    /**
     * @brief mahalaScale is a scaling factor for the Mahalanobis distance visualization.
     * The range [0:x] is marked blue, [x:2x] green, [2x:3x] yellow and >3x red.
     */
    double mahalaScale = 1;

    /**
     * @brief colorFactor normalization factor for flow color visualization
     */
    double colorFactor = 10;

    cv::Vec3b invalid_color = cv::Vec3b(245,219,155); // Sky blue

    /**
     * @brief flow The ground truth flow.
     */
    cv::Mat_<cv::Vec2f> flow;

    /**
     * @brief unc The ground truth uncertainty
     */
    cv::Mat_<cv::Vec2f> unc;

    /**
     * @brief count The ground truth sample count
     */
    cv::Mat density;

    /**
     * @brief name The name of the frame, e.g. "0_0013_001337"
     */
    std::string name;

    /**
     * @brief dynamic contains 1 where the region is marked as dynamic, otherwise 0.
     */
    cv::Mat_<uint8_t> dynamic;

    cv::Mat_<uint8_t> valid;

    cv::Mat_<uint8_t> engineHood;

    std::map<std::string, cv::Mat> algos;

    void addEngineHoodMask(const std::string &dir);

    void writeMasked(const std::string& dir, const double scaling = -1);

    /**
     * Draw all dynamic objects for a given filename on a given image.
     *
     * @param [inout] mask This should be a single-channel black image (value zero), the masks are then drawn with value 1 on it.
     * @param [in] filename The filename for the frame for looking it up in the map of contours.
     * @param [in] contours A map of contours, it is asumed that it has the form std::map<std::string, std::vector<std::vector> > > where the string is the filename, and the vector of vectors is a vector of contours, each contour given by a vector of points.
     */
    template<class Map>
    bool dynamicMask(cv::Mat& mask, std::string const& filename, const Map& contours);

    /**
     * Draw all dynamic objects for a given filename on a given image.
     *
     * @param [inout] mask This should be a single-channel black image (value zero), the masks are then drawn with value 1 on it.
     * @param [in] filename The filename for the frame for looking it up in the map of contours.
     * @param [in] contours A map of contours, it is asumed that it has the form std::map<std::string, std::vector<std::vector> > > where the string is the filename, and the vector of vectors is a vector of contours, each contour given by a vector of points.
     */
    template<class Map>
    bool dynamicMask(std::string const& filename, const Map& contours);

    /**
     * @brief validMask Creates the mask of valid GT pixel.
     * dynamicMask must be called before calling this function.
     */
    void validMask();


    void addAlgo(std::string filename);

    void addAlgos(std::vector<std::string> filenames);

    void write(const std::string& filename);

    static void write(const std::string& filename, const cv::Mat & _flow);

    void write(const std::string& path, const std::string& filename);

    static void write(const std::string& path, const std::string& filename, cv::Mat& _flow);

    /**
     * @brief getSNR calculates the signal-to-noise ratio of the current ground truth.
     * @param snr
     */
    void getSNR(cv::Mat& snr);

    /**
     * @brief getNSR calculates the noise-to-signal ratio of the current ground truth.
     * @param snr
     */
    void getNSR(cv::Mat& snr);

    void getNSRVis(cv::Mat& nsrvis);

    void contourStats();

    static cv::Mat_<cv::Vec3b> colorFlow(const cv::Mat_<cv::Vec2f>& flow, const float factor, const double scaleFactor = -1);

    static cv::Mat grayFlow(const cv::Mat_<cv::Vec2f>& flow, double threshold = -1);

    static double maxFlowLength(const cv::Mat& flow,
            float *_min_u = nullptr,
            float *_max_u = nullptr,
            float *_min_v = nullptr,
            float *_max_v = nullptr);

    void colorFlow(const cv::Mat& flow, const std::string& filename, const double factor, const double scaleFactor = -1);

    template<class Docs>
    /**
     * @brief compare shows the difference between ground truth and the result of some algorithm.
     * @param algo
     */
    void compare(const std::string& algoName, const cv::Mat& algo, const std::string& resultDir, const std::string& resultName, Docs& docs);

    template<class Doc, class String1, class Vector, class Object>
    static void addStats(Doc& doc, String1 key1, QuantileStats<float>& stats, Vector& quantiles, Object& object);

    template<class Doc, class String1, class String2, class String3, class Object>
    static void addDoubleString(Doc& doc, String1 key1, String2 key2, String3 value, Object& object);

    template<class Doc, class String, class Object>
    static void addScore(Doc& doc, String scoreName, double scoreValue, Object& scores);

    template<class Doc, class String, class Object>
    static void addDouble(Doc& doc, String name, double doubleValue, Object& scores);

    template<class Doc, class String1, class String2, class Object>
    static void addString(Doc& doc, String1 key, String2 value, Object& object);

    template<class Doc, class Member, class String, class Object>
    /**
     * @brief addMember Adds a member to an object in a rapidjson::Document
     * @param doc Document which is needed for getting the allocator.
     * @param name Name of the new member added.
     * @param member Member which shall be added.
     * @param object Object within the document to which the member should be added.
     * May also be the document itself.
     */
    static void addMember(Doc& doc, String name, Member& member, Object& object);

    void badpix(const cv::Mat &gt,
            const cv::Mat &submission,
            const cv::Mat &unc,
            const cv::Mat &valid_mask,
            const cv::Mat &fattening_mask,
            RunningStats &result,
            cv::Mat &visu,
            RunningStats &result_fat,
            cv::Mat &visu_fat,
            const double threshold);

    static cv::Size convertSize(cv::MatSize const& src);

    static double flowLength(const cv::Vec2f vec);
};

#endif // FLOWGT_H
