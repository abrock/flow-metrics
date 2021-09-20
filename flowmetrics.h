#ifndef FLOWMETRICS_H
#define FLOWMETRICS_H

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include "contourflow.h"

#include <runningstats/runningstats.h>
using runningstats::RunningCovariance;
using runningstats::QuantileStats;
using runningstats::RunningStats;
using runningstats::BinaryStats;

#include <simplejson/simplejson.hpp>

#include "colors.hpp"

namespace fs=boost::filesystem;


struct FatteningResult {
    /**
     * @brief differenceFlow Difference flow field between algorithm result and reference data (algorithm minus reference)
     */
    cv::Mat_<cv::Vec2f> differenceFlow;
    /**
     * @brief differenceFlowColored Colored version of @see differenceFlow
     */
    cv::Mat_<cv::Vec3b> differenceFlowColored;
    /**
     * @brief colored_result for display on website.
     */
    cv::Mat_<cv::Vec3b> coloredResult;

    cv::Mat_<cv::Vec3b> fineFatteningVisu;

    cv::Mat_<cv::Vec3b> fineThinningVisu;

    /**
     * @brief correlationU Correlation coefficient between the reference data
     * and the difference between reference data and algorithm result.
     */
    RunningCovariance correlationU, correlationV, correlationUV;

    /**
     * @brief corrURaw Raw data of the contribution of each pixel to the correlation coefficient
     */
    cv::Mat_<float> corrURaw, corrVRaw, corrUVRaw;

    /**
     * @brief corrUMax Maximum value of @see corrURaw
     */
    double corrUMax, corrVMax, corrUVMax;

    /**
     * @brief corrMax is the maximum of @see corrUMax, @see corrVMax and @see corrUVMax
     */
    double corrMax = 0;

    /**
     * @brief corrFactor is the normalization factor used for normalizing @see corrURaw, @see corrVRaw and @see corrUVRaw
     * when creating the colored versions.
     */
    double corrFactor = 0;

    /**
     * @brief currUColor colorization of @see corrURaw using the viridis color scale.
     */
    cv::Mat_<cv::Vec3b> corrUColor, corrVColor, corrUVColor;

    BinaryStats edgeFattening;

    BinaryStats fineFattening;
    BinaryStats fineThinning;
    BinaryStats fineErrors;

    BinaryStats invalidPixels;

    /**
     * @brief smoothing Statistics on the strength of the smoothing, @see smoothingData
     */
    QuantileStats<float> smoothing;
    /**
     * @brief smoothingData The parts of the image where the algorithm result is "too smooth"
     */
    cv::Mat_<float> smoothingData;
    /**
     * @brief smoothingVisu Visualization of @see smoothing
     */
    cv::Mat_<cv::Vec3b> smoothingVisu;

    double smoothingVisuScale = -1;

    /**
     * @brief bumpiness Statistics on the strength of the smoothing, @see bumpinessData
     */
    QuantileStats<float> bumpiness;
    /**
     * @brief bumpinessData The parts of the image where the algorithm result is "not smooth enough"
     */
    cv::Mat_<float> bumpinessData;
    /**
     * @brief bumpinessVisu Visualization of @see bumpiness
     */
    cv::Mat_<cv::Vec3b> bumpinessVisu;

    double bumpinessVisuScale = -1;

    /**
     * @brief bumpinessCount Count the number of pixels where bumpiness occurred
     */
    BinaryStats bumpinessCount;

    /**
     * @brief smoothingCount Count the number of pixels where smoothing occurred
     */
    BinaryStats smoothingCount;

    /**
     * @brief endpointErrorStats Classical mean endpoint error stats
     */
    QuantileStats<float> endpointError;

    /**
     * @brief endpointErrorStatsSmooth Classical mean endpoint error stats only at smooth surfaces
     */
    QuantileStats<float> endpointErrorSmooth;

    cv::Mat_<float> endpointErrorData;

    cv::Mat_<float> endpointErrorSmoothData;

    cv::Mat_<cv::Vec3b> endpointErrorVisu;

    cv::Mat_<cv::Vec3b> endpointErrorSmoothVisu;

    QuantileStats<float> gtLength;

    std::string printStats();
};

class FlowMetrics {
    bool callbackLocked = false;

    template<class VAL>
    void setBorder(cv::Mat_<VAL>& mat, int border, VAL const value);

    cv::Mat_<cv::Vec2f> gt;
    cv::Mat grad_x, grad_y, filled_gt, u, v, grad_u_x, grad_u_y, grad_v_x, grad_v_y,
    propagated_growed_gt, unc, filled_unc, propagated_unc, propagated_growed_unc,
    propagated_growed_trimmed_gt,
    colored_propagated_gt,
    colored_propagated_growed_trimmed_gt;

    cv::Mat_<uint8_t> gt_invalid_mask;


    /**
     * @brief ignore_border Ignore a x px border in all evaluations.
     */
    int ignore_border = 5;

    cv::Mat_<float> grad_mag;
    /**
     * @brief segmentation Segmentation of detected edges into connected components
     */
    cv::Mat_<int> segmentation;
    /**
     * @brief filtered_segmentation Segmentation of detected edges into connected components without very small components
     * (number of pixels smaller than #segment_size_threshold)
     */
    cv::Mat_<int> filtered_segmentation;

    /**
     * @brief l2distance2edges exact L2 distances to the edges.
     */
    cv::Mat_<float> l2distance2edges;


    /**
     * @brief l2distance2fine_structures L2 distance to fine structures, @see fine_structures
     */
    cv::Mat_<float> l2distance2fine_structures;



    /**
     * @brief fine_structure_grow_distance Maximum distance between fine structures
     * and evaluation points for fine structure metrics
     */
    double fine_structure_grow_distance = 8;

    /**
     * @brief computeFineStructureNeighbours Compute the mask @see fine_structure_neighbours
     */
    void computeFineStructureNeighbours();

    cv::Mat_<float> grad_mag_disp;
    std::vector<size_t> segment_sizes;
    std::vector<size_t> sizes_hist;
    double grad_mag_threshold = 0.6;
    size_t segment_size_threshold = 40;
    size_t adaptive_block_size = 21;
    double adaptive_delta = -1.5;
    size_t regionGrowingCounter = 9;
    size_t edgeGrowingCounter = 6;
    enum class Showing {
        Gradients,
        Filtered,
        FilteredAndInvalid,
        FatteningSuccess,
        OriginalGT,
        PropagatedGT,
        PropagatedGrowedGT,
        PropagatedGrowedTrimmedGT,
        PropagatedGrowedTrimmedDistinguishableGT,
        AlgoResult,
        AlgoFattening
    };
    Showing show = Showing::Gradients;
    double max_dist = 5;
    int propagation_window_size = static_cast<int>(regionGrowingCounter);

    /**
     * @brief fattening_success Points next to GT edges where the background/foreground propagation was successfull.
     */
    cv::Mat_<uint8_t> fattening_success;

    /**
     * @brief propagated_gt Matrix with the propagated ground truth from across the edges.
     */
    cv::Mat propagated_gt;

    cv::Mat_<uint8_t> direct_distance;

    bool needs_recalc = true;


    /**
     * @brief similarity_threshold For edge fattening
     */
    double similarity_threshold = 1;

    /**
     * @brief save_all_images Set to true if all images should be saved to disk.
     */
    bool save_all_images = false;


    cv::Vec3b good_color = ColorsRD::yellow();

    cv::Vec3b bad_color = ColorsRD::red();

    FatteningResult res;


    /**
     * @brief same_plane_threshold When fitting a homography to a part of the ground truth flow we have to decide
     * if the fit is "good enough". We calculate the maximum of the euclidean distance of the residuals and if it is
     * below this threshold we assume that the pixels belong to the same plane.
     */
    double same_plane_threshold = .1;


    /**
     * @brief fine_structure_threshold The radius of the fine structure detection window. The actual width/height is 2*n+1
     */
    int fine_structure_threshold = 9;

    cv::Point const eight_neighbours [8] = {
        cv::Point(-1, -1),
        cv::Point(+1, +1),
        cv::Point(-1, +1),
        cv::Point(+1, -1),
        cv::Point(-1,  0),
        cv::Point(+1,  0),
        cv::Point( 0, -1),
        cv::Point( 0, +1)
    };

    cv::Point nine_neighbours [9];
    cv::Point twentyfive_neighbours [25];
    cv::Point fourtynine_neighbours [49];

    std::vector<cv::Vec3b> color_cache;

    /**
     * @brief min_neighbour_neighbour_dist The minimum possible distance between
     * two sets of points on opposing sides of a fine structure is
     * 1px edge + 1px fine structure + 1px edge + 1px = 4px.
     * Any two regions which are closer than 4px can not have a fine structure
     * between them.
     */
    double min_neighbour_neighbour_dist = 4;

    /**
     * @brief max_neighbour_dist The maximum distance between two regions separated by an edge.
     * if the regions are 4px or more apart there might be a fine structure between them.
     */
    double max_neighbour_dist = 3.99;


    size_t unfilteredEdgeGrowingCounter = 3;

    cv::Mat_<float> l2distance2unfiltered_edges;

public:
    cv::Mat_<uint8_t> smoothing_mask;

    cv::Mat propagated_growed_trimmed_distinguishable_gt;
    cv::Mat propagated_growed_trimmed_distinguishable_unc;


    cv::Mat_<cv::Vec3b> colored_algo;

    double color_scale = 5;

    /**
     * @brief fine_detection_process_visu Visualization of the process of finding fine structures.
     * white: no fine structure
     * black: edge
     * yellow: fine structure detected by @see findFineStructureSimple
     * green: fine structure detected by both findFineStructureSimple and @see findFineStructureComplex
     */
    cv::Mat_<cv::Vec3b> fine_detection_process_visu;

    /**
     * @brief fine_detection_simple_visu Visualization of the fine structures detected by @see findFineStructureSimple
     * white: no fine structure
     * black: edge
     * green: fine structure detected by @see findFineStructureSimple
     */
    cv::Mat_<cv::Vec3b> fine_detection_simple_visu;

    cv::Mat_<cv::Vec3b> colored_gt;
    cv::Mat_<cv::Vec3b> colored_propagated_growed_trimmed_distinguishable_gt;
    cv::Mat_<cv::Vec3b> colored_propagated_growed_gt;
    cv::Mat_<cv::Vec3b> colored_propagated_growed_gt_edges;

    cv::Mat_<cv::Vec3b> filled_gt_visu;
    cv::Mat_<cv::Vec3b> grad_mag_visu;
    cv::Mat_<cv::Vec3b> adaptive_threshold_visu;
    cv::Mat_<uint8_t> unfiltered_edges;

    cv::Mat_<cv::Vec3b> colored_propagated_growed_gt_with_edges;

    cv::Mat_<uint8_t> unfiltered_edges_grown;

    cv::Mat_<cv::Vec3b> fine_structure_mask_visu;

    /**
     * @brief filtered_edge_mask_grown variant of filtered_edge_mask where the edge regions have been grown by edgeGrowingCounter pixels.
     */
    cv::Mat_<uint8_t> filtered_edge_mask_grown;

    /**
     * @brief filtered_edge_mask zero where no edge is, 255 where an edge was detected and the size of the connected region
     * is large enough.
     */
    cv::Mat_<uint8_t> filtered_edge_mask;

    /**
     * @brief filtered_edge_mask zero where no edge is, 255 where an edge was detected and the size of the connected region
     * is large enough.
     */
    cv::Mat_<uint8_t> filtered_edge_mask_no_border;

    cv::Mat_<cv::Vec3b> l2distance2edges_visu;

    /**
     * @brief fine_structure_neighbours Mask of the regions within a certain distance to fine structures
     */
    cv::Mat_<uint8_t> fine_structure_neighbours;

    /**
     * @brief fine_structures Mask for fine structure evaluation: 255 where a fine structure was detected, 0 everywhere else
     */
    cv::Mat_<uint8_t> fine_structures;

    void testPlane();

    template<class Flow>
    double maxError(Flow const& flow, std::vector<cv::Point2d> const& src, std::vector<cv::Point2d> const& dst) const;

    template<class Flow>
    double flowError(Flow const& flow, cv::Point2d const& src, cv::Point2d const& dst) const;

    static void plotHist(cv::Mat const& src, const std::string filename, int num_bins = 100);

    void safeAllImages();

    /**
     * @brief algo Algorithm result;
     */
    cv::Mat_<cv::Vec2f> algo;

    FlowMetrics(cv::Mat const& _gt, double uncertainty = 1);

    FlowMetrics();

    void evaluate(const std::string & gt_id,
            cv::Mat const & gt,
            cv::Mat const & submission, const cv::Mat &unc,
            const boost::filesystem::path &dst_dir,
            json::JSON & json,
            json::JSON & metrics_definitions);

    cv::Mat get_propagated_growed_gt();

    static cv::Mat get_propagated_growed_gt(cv::Mat const& _gt);

    cv::Mat get_edges();

    static cv::Mat get_edges(cv::Mat const& _gt);

    void mouseCallBack(int event, int x, int y, int flags);

    template<class MASK>
    void findInvalidFlow(cv::Mat const& src, cv::Mat_<MASK>& mask, MASK const invalid_val = std::numeric_limits<MASK>::max());

    template<class SEG, class MASK>
    void filterSegments(
            cv::Mat_<SEG> const& src,
            cv::Mat_<SEG> & dst,
            cv::Mat_<MASK> & dst_mask,
            std::vector<size_t> const& sizes,
            size_t const threshold);

    template<class MASK>
    void filterSegments(
            cv::Mat_<MASK> const& src,
            cv::Mat_<MASK> & dst,
            size_t const threshold);

    template<class SEG, class MASK>
    void closeSegmentGaps(cv::Mat_<SEG>const & src, cv::Mat_<SEG>& dst, cv::Mat_<MASK>& mask);


    template<class SRC, class SEG>
    size_t floodFill8(cv::Mat_<SRC> const& src, cv::Mat_<SEG> & seg, int ii, int jj, size_t value);

    template<class SRC, class SEG, class SIZE>
    void getSegmentation(cv::Mat_<SRC> const& src, cv::Mat_<SEG>& segmentation, std::vector<SIZE> & segment_sizes);

    void init(cv::Mat const& _gt, double const uncertainty = 1);


    void init(cv::Mat const& _gt, cv::Mat const& _unc);

    bool isValidIndex(cv::Point const p, cv::Mat const& mat);

    template<class MASK, class DIST>
    void distanceMeasurement(cv::Mat_<MASK> const& edges,
                             cv::Mat_<DIST> & distances,
                             bool cross_edges,
                             int const ii,
                             int const jj,
                             int const propagation_window_size);

    template<class MASK>
    bool localFatteningPropagation(
            cv::Mat const& gt,
            cv::Mat_<MASK> const& edges,
            cv::Mat & dst,
            cv::Mat const& unc,
            cv::Mat& unc_dst,
            int const ii, int const jj,
            cv::Mat_<cv::Vec3b>* colors = nullptr);

    void removeInvalidFlow(cv::Mat const& orig_gt, cv::Mat& propagated);

    bool is_distinguishable(
            double const a_val, double const a_unc,
            double const b_val, double const b_unc,
            double const threshold = 1);

    bool is_distinguishable(
            cv::Vec2f const a_flow, cv::Vec2f const a_unc,
            cv::Vec2f const b_flow, cv::Vec2f const b_unc,
            double const threshold = 1);

    /**
     * @brief remove_similar_bg_fg Removes parts of the propagated ground truth used for measuring edge fattening
     * where the ground truth and the propagated ground truth are so similar that evaluation makes no sense
     * given the ground truth uncertainty.
     * @param gt Original ground truth
     * @param unc Original uncertainties
     * @param prop_gt Propagated ground truth
     * @param prop_unc Propagated uncertainties
     */
    void remove_similar_bg_fg(
            cv::Mat const& gt, cv::Mat const& unc,
            cv::Mat& prop_gt, cv::Mat& prop_unc,
            double const threshold = 1);

    template<class MASK>
    void regionGrowing(
            cv::Mat const& orig_src,
            cv::Mat& dst,
            cv::Mat const& orig_unc,
            cv::Mat& dst_unc,
            cv::Mat_<MASK> const& edges);

    template<class MASK>
    void fatteningPropagation(cv::Mat const& gt, cv::Mat_<MASK> const& edges, cv::Mat & dst, cv::Mat const& unc, cv::Mat& unc_dst);

    void run_propagation(cv::Mat const& _gt);

    void run(cv::Mat const& _gt, double const scalar_unc = 1);

    void run(cv::Mat const& _gt, cv::Mat const& _unc);

    static void normalize_by_flow_length(cv::Mat const& flow, cv::Mat& srcdst, double epsilon = 1);

    void prepare_images();

    void showImg();

    FatteningResult evaluateFattening(cv::Mat const& _algo);

    void bilateralFlowFilter(cv::Mat &src, cv::Mat &dst, int d, double sigmaColor, double sigmaSpace, int borderType = cv::BORDER_DEFAULT);

    template<class MASK>
    void findPlanes(cv::Mat const& flow, cv::Mat_<MASK> const& edges);

    template<class MASK>
    bool findPlane(cv::Mat const& flow, cv::Mat_<MASK> const& edges, cv::Point2i const initial);

    template<class MASK, class FLOW>
    bool findArea(cv::Mat const& flow, cv::Mat_<MASK> const& edges, cv::Point2i const initial);

    void testPlane2();

    /**
     * @brief evaluateSimple evaluates all metrics given a GT flow field, a submission flow field,
     * etc.
     * @param submission_filename Filename of the submission, is used as prefix for all result images.
     * @param gt Reference data flow field
     * @param submission
     * @param unc
     * @param dst_dir
     * @param json
     * @param metrics_definitions
     */
    void evaluateSimple(const std::string &visualization_prefix,
            const cv::Mat &gt,
            const cv::Mat &submission,
            const cv::Mat &unc,
            json::JSON &json,
                        json::JSON &metrics_definitions);

    template<class SEG>
    /**
     * @brief findFineStructure Checks if a given pixel in an image of detected edges corresponds to a fine structure.
     * @param edges Image of edges (!=0) and smooth sections (==0)
     * @param center pixel which should be checked
     * @param debug set this to true if you want debug images shown and debug output generated.
     * @return true if a fine structure was detected.
     */
    bool findFineStructure(
            const cv::Mat_<SEG> &edges,
            const cv::Point center,
            bool const debug = false);

    template<class SEG>
    /**
     * @brief benchmarkFindFineStructure
     * @param edges
     * @param center
     * @param debug
     * @return
     */
    bool benchmarkFindFineStructure(
            const cv::Mat_<SEG> &edges,
            const cv::Point center,
            bool const debug = false);

    template<class SEG>
    cv::Mat_<cv::Vec3b> colorSegmentation(cv::Mat_<SEG> const& segmentation);

    template<class SEG, class STRUCT>
    void findFineStructureGlobal(
            cv::Mat_<SEG> const& edges,
            cv::Mat_<STRUCT> & result,
            bool debug = false);

    /**
     * @brief naiveNearestNeighbour Find the smallest distance between two sets of points.
     * @param A set one
     * @param B set two
     * @param stop_if_smaller_than additional criterion for stopping.
     * If a pair with smaller or equal distance is found the algorithm terminates
     * and returns the distance of this pair regardless of possible other smaller distances.
     * @return The smallest distance found before the stopping criterion as fulfilled.
     * A stopping criterion of zero will make the function check every pair
     * until either all pairs have been checked or one pair had distance zero.
     */
    static double naiveNearestNeighbour(
            std::vector<cv::Point> const& A,
            std::vector<cv::Point> const& B,
            double const stop_if_smaller_than = 0);

    template<class SEG>
    bool findFineStructureComplex(
            const cv::Mat_<SEG> &edges,
            const cv::Point center,
            const bool debug);

    template<class SEG>
    bool findFineStructureCombined(
            const cv::Mat_<SEG> &edges,
            const cv::Point center,
            const bool debug);

    template<class SEG, class STRUCT>
    void closeStructureHoles(const cv::Mat_<SEG> &edges, cv::Mat_<STRUCT> &structures);

    template<class SEG, class STRUCT>
    void paintCircleToEdge(
            const cv::Mat_<SEG> &edges,
            cv::Mat_<STRUCT> &result,
            const cv::Point &center,
            const int window_rad);

    template<class SEG>
    /**
     * @brief edgeDistance Measures the distance between an point and the nearest edge pixel along a given direction.
     * @param edges
     * @param center
     * @param direction
     * @param max_dist
     * @return
     */
    double edgeDistance(
            const cv::Mat_<SEG> &edges,
            const cv::Point &center,
            const cv::Point2f &direction,
            const double max_dist);

    template<class SEG>
    /**
     * @brief findFineStructureSimple Tells if a pixel belong to a fine structure by checking if
     * there are edges within a small distance in two opposing directions
     * @param edges
     * @param center
     * @param max_dist
     * @return
     */
    bool findFineStructureSimple(const cv::Mat_<SEG> &edges, const cv::Point center, const double max_dist);
    template<class SEG>
    /**
     * @brief make4neighbours8neighbours searches for 4-connected segments of nonzero elements and makes them 8-connected.
     * @param data
     */
    cv::Mat_<SEG> make4neighbours8neighbours(cv::Mat_<SEG> const & src);
};



#endif // FLOWMETRICS_H
