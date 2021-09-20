#include "flowmetrics.h"
#include "flowgt.h"
#include <stack>
#include <vector>
#include <map>
#include <sstream>
#include <simplejson/simplejson.hpp>

#include "metric-helpers.h"

#include "gnuplot-iostream.h"

#include <boost/algorithm/clamp.hpp>

#include <bitset>

#include "colors.hpp"

#include <colorscales/colorscales.h>

void FlowMetrics::mouseCallBack(int event, int x, int y, int flags) {
    ParallelTime execution_time;
    if (callbackLocked) {
        return;
    }
    bool success = false;
    callbackLocked = true;

    // is set to true if anything relevant happens.
    bool any_action = false;
    bool const ctrl_pressed = flags & (1<<3);
    bool const shift_pressed = flags & (1<<4);
    int const ii = y;
    int const jj = x;
    int const rows = gt.rows;
    int const cols = gt.cols;
    cv::Point2i const clicked_point(x, y);
    std::string chosen_action = "none";
    if (cv::EVENT_MBUTTONDOWN == event) { // Test the findArea method starting at the clicked point.
        chosen_action = "FlowMetrics::findArea<uint8_t, HomographyFlow>(gt, filtered_seg_mask, cv::Point2i(x, y));";
        success = FlowMetrics::findArea<uint8_t, HomographyFlow>(gt, filtered_edge_mask_grown, cv::Point2i(x, y));
        any_action = true;
    }
    if (cv::EVENT_RBUTTONDOWN == event) {
        any_action = true;

        if (ctrl_pressed) {
            if (shift_pressed) {
                /*
                chosen_action = "FlowMetrics::findArea<uint8_t, Polynome3Flow>(gt, filtered_seg_mask, cv::Point2i(x, y));";
                bool success = FlowMetrics::findArea<uint8_t, Polynome3Flow>(gt, filtered_edge_mask_grown, cv::Point2i(x, y));
                std::cout << (success ? "success :-)" : "no success :-(") << std::endl;
                */
                chosen_action = "findFineStructure";
                success = findFineStructureCombined(filtered_edge_mask, clicked_point, true);
                findFineStructureComplex(filtered_edge_mask, clicked_point, true);
                std::cout << (success ? "success :-)" : "no success :-(") << std::endl;
                //benchmarkFindFineStructure(filtered_edge_mask, clicked_point, true);
                findFineStructureGlobal(filtered_edge_mask, fine_structures, true);
                cv::namedWindow("detected fine structures");
                cv::imshow("detected fine structures", fine_structures);
            }
            else {
                chosen_action = "FlowMetrics::findArea<uint8_t, Polynome1Flow>(gt, filtered_seg_mask, cv::Point2i(x, y));";
                success = FlowMetrics::findArea<uint8_t, Polynome1Flow>(gt, filtered_edge_mask_grown, cv::Point2i(x, y));
                std::cout << (success ? "success :-)" : "no success :-(") << std::endl;
            }
        }
        else if (shift_pressed) {
            chosen_action = "FlowMetrics::findArea<uint8_t, Polynome2Flow>(gt, filtered_seg_mask, cv::Point2i(x, y));";
            success = FlowMetrics::findArea<uint8_t, Polynome2Flow>(gt, filtered_edge_mask_grown, cv::Point2i(x, y));
            std::cout << (success ? "success :-)" : "no success :-(") << std::endl;
        }
        else {
            chosen_action = "FlowMetrics::findArea<uint8_t, Polynome0Flow>(gt, filtered_seg_mask, cv::Point2i(x, y));";
            success = FlowMetrics::findArea<uint8_t, Polynome0Flow>(gt, filtered_edge_mask_grown, cv::Point2i(x, y));
            std::cout << (success ? "success :-)" : "no success :-(") << std::endl;
        }

        std::cout << "x: " << x << ", y: " << y << std::endl;
        std::cout << "filled_gt: " << filled_gt.at<cv::Vec2f>(y, x) << std::endl;
        std::cout << "grad_mag: " << grad_mag.at<float>(y, x) << std::endl;
        std::cout << "grad_x: " << grad_x.at<cv::Vec2f>(y, x) << std::endl;
        std::cout << "grad_y: " << grad_y.at<cv::Vec2f>(y, x) << std::endl;
        std::cout << "u: " << u.at<float>(y, x) << std::endl;
        std::cout << "grad_u_x: " << grad_u_x.at<float>(y, x) << std::endl;
        std::cout << "grad_u_y: " << grad_u_y.at<float>(y, x) << std::endl;
        std::cout << "segmentation: " << segmentation.at<int>(y, x) << std::endl;
        std::cout << "l2distance2edges: " << l2distance2edges(y, x) << std::endl;
        if (segment_sizes.size() > size_t(segmentation.at<int>(y, x))) {
            std::cout << "segment size: " << segment_sizes[segmentation.at<int>(y, x)] << std::endl;
        }
    }
    /*
        if (!segment_sizes.empty()) {
            sizes_hist.clear();
            for (auto const size : segment_sizes) {
                while (sizes_hist.size() <= size) {
                    sizes_hist.push_back(0);
                }
                sizes_hist[size]++;
            }
            for (size_t ii = 0; ii < sizes_hist.size(); ++ii) {
                if (sizes_hist[ii] > 0) {
                    std::cout << "seg size " << ii << " occured " << sizes_hist[ii] << " times" << std::endl;
                }
            }
        }
        // */
    if (cv::EVENT_LBUTTONDOWN == event) {
        any_action = true;
        cv::Mat_<uint8_t> indirect_distance(2*propagation_window_size + 1, 2*propagation_window_size + 1, 255);
        indirect_distance(propagation_window_size, propagation_window_size) = 0;
        distanceMeasurement(filtered_edge_mask, indirect_distance, false, ii, jj, propagation_window_size);
        distanceMeasurement(filtered_edge_mask, direct_distance, true, ii, jj, propagation_window_size);

        std::cout << "direct distances:" << std::endl
                  << direct_distance << std::endl;
        std::cout << "indirect distances:" << std::endl
                  << indirect_distance << std::endl;

        cv::Mat_<cv::Vec3b> colors(rows, cols, cv::Vec3b(50,50,50));
        cv::cvtColor(filtered_edge_mask, colors, cv::COLOR_GRAY2BGR);
        cv::Mat dst(rows, cols, CV_32FC2);

        //bool localFatteningPropagation(cv::Mat const& gt, cv::Mat_<MASK> const& edges, cv::Mat & dst, int const ii, int const jj, cv::Mat_<cv::Vec3b>* colors = nullptr) {

        chosen_action = "localFatteningPropagation(gt, filtered_seg_mask, unc, propagated_unc, dst, ii, jj, &colors);";
        localFatteningPropagation(gt, filtered_edge_mask, unc, propagated_unc, dst, ii, jj, &colors);

        cv::imshow("magnitudes", colors);

    }
    if (any_action) {
        std::cout << "Event was: " << event << ", flags: " << std::bitset<32>(flags) << std::endl;
        std::cout << "Chosen action: " << chosen_action << std::endl;
        std::cout << (success ? "success :-)" : "no success :-(") << std::endl;
        std::cout << "Execution time: " << execution_time.print() << std::endl;
    }
    callbackLocked = false;
}

template<class SEG>
cv::Mat_<SEG> FlowMetrics::make4neighbours8neighbours(cv::Mat_<SEG> const & data) {
    cv::Mat_<SEG> src = data.clone();
    cv::Mat_<SEG> dst = data.clone();

    bool did = true;
    while (did) {
        did = false;
        for (int ii = 1; ii < src.rows; ++ii) {
            for (int jj = 1; jj < src.cols; ++jj) {
                cv::Point const top_left(jj-1, ii-1);
                cv::Point const top_right(jj, ii-1);
                cv::Point const bottom_left(jj-1, ii);
                cv::Point const bottom_right(jj, ii);
                if (
                        src(top_left) != 0 &&
                        src(bottom_right) != 0 &&
                        src(top_right) == 0 &&
                        src(bottom_left) == 0) {
                    dst(top_right) = dst(bottom_left) = src(top_left);
                    did = true;
                }
                if (
                        src(bottom_left) != 0 &&
                        src(top_right) != 0 &&
                        src(bottom_right) == 0 &&
                        src(top_left) == 0) {
                    dst(bottom_right) = dst(top_left) = src(bottom_left);
                    did = true;
                }
            }
        }
        src = dst.clone();
    }
    return dst;
}

template<class SEG, class STRUCT>
void FlowMetrics::closeStructureHoles(
        cv::Mat_<SEG> const& edges,
        cv::Mat_<STRUCT>& structures) {
    bool success = true;
    size_t global_iterations = 0;
    for (global_iterations = 0; global_iterations < 100 && success; ++global_iterations) {
        global_iterations++;
        success = false;
        for (int ii = 1; ii+1 < structures.rows; ++ii) {
            for (int jj = 1; jj+1 < structures.cols; ++jj) {
                cv::Point const center(jj, ii);
                if (structures(center) == 0 && edges(center) == 0) {
                    cv::Point const left(jj-1, ii);
                    cv::Point const right(jj+1, ii);
                    if (structures(left) != 0 && structures(right) != 0) {
                        structures(center) = structures(left);
                        success = true;
                        continue;
                    }
                    cv::Point const top(jj, ii-1);
                    cv::Point const bottom(jj, ii+1);
                    if (structures(top) != 0 && structures(bottom) != 0) {
                        structures(center) = structures(top);
                        success = true;
                        continue;
                    }
                    cv::Point const topleft(jj-1, ii-1);
                    cv::Point const bottomright(jj+1, ii+1);
                    if (structures(topleft) != 0 && structures(bottomright) != 0) {
                        structures(center) = structures(topleft);
                        success = true;
                        continue;
                    }
                    cv::Point const topright(jj+1, ii-1);
                    cv::Point const bottomleft(jj-1, ii+1);
                    if (structures(topright) != 0 && structures(bottomleft) != 0) {
                        structures(center) = structures(topright);
                        success = true;
                        continue;
                    }
                    size_t structure_neighbour_counter = 0;
                    size_t edge_neighbour_counter = 0;
                    for (cv::Point const& neighbour_offset : eight_neighbours) {
                        cv::Point const neighbour = center + neighbour_offset;
                        if (structures(neighbour) != 0) {
                            structure_neighbour_counter++;
                        }
                        if (edges(neighbour) != 0) {
                            edge_neighbour_counter++;
                        }
                    }
                    if (structure_neighbour_counter >= 4) {
                        structures(center) = STRUCT(255);
                        continue;
                    }
                    if (structure_neighbour_counter >= 3 && edge_neighbour_counter >= 3) {
                        structures(center) = STRUCT(255);
                        continue;
                    }
                }
            }
        }
    }

}

template<class SEG>
double FlowMetrics::edgeDistance(
        cv::Mat_<SEG> const& edges,
        cv::Point const& center,
        cv::Point2f const& direction,
        double const max_dist) {

    for (double dist = .5; dist <= max_dist; dist += .5) {
        cv::Point2i const offset = cv::Point2i(
                    static_cast<int>(std::trunc(dist * static_cast<double>(direction.x))),
                    static_cast<int>(std::trunc(dist * static_cast<double>(direction.y))));
        cv::Point2i const current_point = center + offset;
        if (!isValidIndex(current_point, edges)) {
            return std::numeric_limits<double>::max();
        }
        if (edges(current_point) != 0) {
            return std::sqrt(offset.dot(offset));
        }
    }
    return std::numeric_limits<double>::max();
}

#if 0
template<class SEG>
bool FlowMetrics::findFineStructureSimple(
        cv::Mat_<SEG> const& edges,
        cv::Point const center,
        double const max_dist) {
    if (edges(center) != 0) {
        return false;
    }
    size_t const num_directions = 4;
    for (size_t ii = 0; ii < num_directions; ++ii) {
        double const angle = M_PI * static_cast<double>(ii) / num_directions;
        cv::Point2f direction(std::cos(angle), std::sin(angle));
        double const dist_a = edgeDistance(edges, center, direction, max_dist);
        if (dist_a <= max_dist) {
            direction *= -1;
            double const dist_b = edgeDistance(edges, center, direction, max_dist);
            if (dist_b <= max_dist) {
                return true;
            }
        }
    }
    return false;
}

#else
template<class SEG>
bool FlowMetrics::findFineStructureSimple(
        cv::Mat_<SEG> const& edges,
        cv::Point const center,
        double const max_dist) {
    if (edges(center) != 0) {
        return false;
    }
    size_t const num_directions = 4;
    for (size_t ii = 0; ii < num_directions; ++ii) {
        double const angle = M_PI * static_cast<double>(ii) / num_directions;
        cv::Point2f direction(std::cos(angle), std::sin(angle));
        double const dist_a = edgeDistance(edges, center, direction, 2*max_dist);
        direction *= -1;
        double const dist_b = edgeDistance(edges, center, direction, 2*max_dist);
        if (dist_b + dist_a <= 2*max_dist) {
            return true;
        }
    }
    return false;
}
#endif

template<class SEG, class STRUCT>
void FlowMetrics::paintCircleToEdge(
        cv::Mat_<SEG> const& edges,
        cv::Mat_<STRUCT> & result,
        cv::Point const& center,
        int const window_rad) {

    int sq_dist_nearest_edge = std::numeric_limits<int>::max();
    bool success = false;
    for (int dii = -window_rad; dii < window_rad; ++dii) {
        for (int djj = -window_rad; djj < window_rad; ++djj) {
            cv::Point const offset(dii, djj);
            cv::Point const current = center + offset;
            if (isValidIndex(current, edges)) {
                if (edges(current) != 0) {
                    int const current_sq_dist = dii * dii + djj * djj;
                    if (sq_dist_nearest_edge > current_sq_dist) {
                        sq_dist_nearest_edge = current_sq_dist;
                        success = true;
                    }
                }
            }
        }
    }
    if (!success) {
        return;
    }
    for (int dii = -window_rad; dii < window_rad; ++dii) {
        for (int djj = -window_rad; djj < window_rad; ++djj) {
            int const current_sq_dist = dii * dii + djj * djj;
            cv::Point const offset(dii, djj);
            cv::Point const current = center + offset;
            if (isValidIndex(current, edges)) {
                if (edges(current) == 0) {
                    if (sq_dist_nearest_edge >= current_sq_dist) {
                        result(current) = STRUCT(255);
                    }
                }
            }
        }
    }
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    //std::cout << "Point clicked: " << x << ", " << y << std::endl;
    static_cast<FlowMetrics*>(userdata)->mouseCallBack(event, x, y, flags);
}




void FlowMetrics::plotHist(const cv::Mat &src, std::string const filename, int num_bins) {
    double min = 0, max = 0;
    cv::minMaxLoc(src, &min, &max);
    max += std::max(1e-10, 1e-7*(max - min));
    int channels[] = {0};
    int histsize[] = {num_bins};
    float range1[] = {static_cast<float>(min), static_cast<float>(max)};
    float const * ranges[] = {range1};
    cv::Mat cv_hist;
    cv::calcHist(&src, 1, channels, cv::Mat(), cv_hist, 1, histsize, ranges);
    //std::cout << cv_hist << std::endl;
    std::vector<std::pair<float, float> > hist;
    for (int ii = 0; ii < num_bins; ++ii) {
        float const bin_val = cv_hist.at<float>(ii);
        float const bin_left = min + (max-min)*ii / num_bins;
        hist.push_back(std::pair<float, float>(bin_left, bin_val));
    }
    gnuplotio::Gnuplot plt;
    std::stringstream command;
    command << "set term svg\n"
            << "set output '" << filename << ".svg' \n"
            << "set title 'histogram' \n"
            << "plot " << plt.file1d(hist, filename + ".hist-data") << " w boxes notitle";

    plt << command.str();
    std::ofstream out(filename + ".gpl");
    out << command.str();
}

void FlowMetrics::safeAllImages() {
    save_all_images = true;
}

FlowMetrics::FlowMetrics(cv::Mat const& _gt, double uncertainty) {
    init(_gt, uncertainty);
}

void makeNeighbours(cv::Point data[], int size) {
    size_t index = 0;
    for (int yy = -size; yy <= size; ++yy) {
        for (int xx = -size; xx <= size; ++xx) {
            data[index] = cv::Point(xx, yy);
            index++;
        }
    }
}

FlowMetrics::FlowMetrics() {}


// gt_id, gt, submission, unc, dst_dir, json_result, metrics_definitions
void FlowMetrics::evaluate(
        const std::string & gt_id,
        cv::Mat const & gt,
        cv::Mat const & submission,
        cv::Mat const & unc,
        const boost::filesystem::path &dst_dir,
        json::JSON & json,
        json::JSON & metrics_definitions) {

    if (submission.cols != gt.cols || submission.rows != gt.rows) {
        throw std::runtime_error("Submission and GT sizes don't match: " + GT::matsize(submission) + " vs. " + GT::matsize(gt));
    }
    init(gt, unc);
    FatteningResult result = evaluateFattening(submission);

    std::string fattening_visu_filename = gt_id + "-fattening-colored.png";
    cv::imwrite((dst_dir / fattening_visu_filename).string(), result.coloredResult);

    json[gt_id]["scores"]["edge-fattening"]["value"] = result.edgeFattening.getPercent();
    json[gt_id]["scores"]["edge-fattening"]["visualization"]["thumb"] = fattening_visu_filename;

    metrics_definitions["metrics"]["edge_fattening"] = {"category", "General",
            "description", "The percentage of pixels near flow edges where the algorithm result is closer to the ground truth on the other side of the edge.",
            "display_name", "EF"};
    metrics_definitions["metrics"]["edge_fattening"]["result_table_visualization"] = {
            "is_visible", true,
            "legend", "yellow = correct, red = wrong, gray = flow length of result"};

}

void showViridisScale(std::string const prefix, int horizontal, int vertical, int thickness = 35) {
    cv::Mat_<float> horiz(thickness, horizontal);
    for (int ii = 0; ii < horiz.rows; ++ii) {
        for (int jj = 0; jj < horiz.cols; ++jj) {
            horiz(ii, jj) = jj;
        }
    }

    cv::Mat_<float> vert(vertical, thickness);
    for (int ii = 0; ii < vert.rows; ++ii) {
        for (int jj = 0; jj < vert.cols; ++jj) {
            vert(ii, jj) = -ii;
        }
    }

    cv::Mat colored = ColorScales::colorViridisAutoscale(horiz);
    cv::imwrite(prefix + "-viridis-horizontal.png", colored);

    colored = ColorScales::colorViridisAutoscale(vert);
    cv::imwrite(prefix + "-viridis-vertical.png", colored);
}

void FlowMetrics::evaluateSimple(
        const std::string & visualization_prefix,
        cv::Mat const & gt,
        cv::Mat const & submission,
        cv::Mat const & unc,
        json::JSON & json,
        json::JSON & metrics_definitions) {

    showViridisScale(visualization_prefix, submission.cols, submission.rows);

    if (submission.cols != gt.cols || submission.rows != gt.rows) {
        throw std::runtime_error("Submission and GT sizes don't match: " + GT::matsize(submission) + " vs. " + GT::matsize(gt));
    }
    std::vector<double> const quantiles = {0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99};
    std::vector<double> const badpix = {1,2,5,10};
    init(gt, unc);
    FatteningResult result = evaluateFattening(submission);

    // Edge fattening
    std::string const fattening_visu_filename = visualization_prefix + "-fattening-colored.png";
    cv::imwrite(fattening_visu_filename, result.coloredResult);

    json[visualization_prefix]["scores"]["edge-fattening"]["value"] = result.edgeFattening.getPercent();
    json[visualization_prefix]["scores"]["edge-fattening"]["visualization"]["large"] = fattening_visu_filename;

    // Fine Fattening
    std::string const fine_fattening_filename = visualization_prefix + "-fine-fattening-colored.png";
    cv::imwrite(fine_fattening_filename, result.fineFatteningVisu);
    json[visualization_prefix]["scores"]["fine-fattening"]["value"] = result.fineFattening.getPercent();
    json[visualization_prefix]["scores"]["fine-fattening"]["visualization"]["large"] = fine_fattening_filename;

    // Fine Thinning
    std::string const fine_thinning_filename = visualization_prefix + "-fine-thinning-colored.png";
    cv::imwrite(fine_thinning_filename, result.fineThinningVisu);
    json[visualization_prefix]["scores"]["fine-thinning"]["value"] = result.fineThinning.getPercent();
    json[visualization_prefix]["scores"]["fine-thinning"]["visualization"]["large"] = fine_thinning_filename;

    // Bumpiness
    std::string const bumpiness_filename = visualization_prefix + "-bumpiness-colored.png";
    cv::imwrite(bumpiness_filename, result.bumpinessVisu);
    json[visualization_prefix]["scores"]["bumpiness"]["value"] = result.bumpiness.getMean();
    json[visualization_prefix]["scores"]["bumpiness"]["max_value"] = result.bumpiness.getMax();
    json[visualization_prefix]["scores"]["bumpiness"]["visualization"]["large"] = bumpiness_filename;
    json[visualization_prefix]["scores"]["bumpiness"]["visualization"]["normalization"] = res.bumpinessVisuScale;

    // Smoothing
    std::string const smoothing_filename = visualization_prefix + "-smoothing-colored.png";
    cv::imwrite(smoothing_filename, result.smoothingVisu);
    json[visualization_prefix]["scores"]["smoothing"]["value"] = result.smoothing.getMean();
    json[visualization_prefix]["scores"]["smoothing"]["max_value"] = result.smoothing.getMax();
    json[visualization_prefix]["scores"]["smoothing"]["visualization"]["large"] = smoothing_filename;
    json[visualization_prefix]["scores"]["smoothing"]["visualization"]["normalization"] = res.smoothingVisuScale;


    // Difference
    std::string const difference_filename = visualization_prefix + "-difference.png";
    cv::imwrite(difference_filename, result.differenceFlowColored);
    json[visualization_prefix]["scores"]["difference"]["visualization"]["large"] = difference_filename;

    // Correlation of gt_u and diff_u
    json[visualization_prefix]["scores"]["correlation_u"]["value"] = result.correlationU.getCorr();
    json[visualization_prefix]["scores"]["correlation_u"]["desc"] = "The correlation coefficient between the ground truth u channel and the (algorithm - gt) u channel.";
    json[visualization_prefix]["scores"]["correlation_u"]["%_eval"] = GT::percent(result.correlationU.getN(), gt);
    std::string const corr_u_filename = visualization_prefix + "-corr-u.png";
    cv::imwrite(corr_u_filename, result.corrUColor);
    json[visualization_prefix]["scores"]["correlation_u"]["visualization"]["large"] = corr_u_filename;
    json[visualization_prefix]["scores"]["correlation_u"]["visualization"]["max_val"] = result.corrUMax;
    json[visualization_prefix]["scores"]["correlation_u"]["visualization"]["normalization"] = result.corrFactor;

    // Correlation of gt_v and diff_v
    json[visualization_prefix]["scores"]["correlation_v"]["value"] = result.correlationV.getCorr();
    json[visualization_prefix]["scores"]["correlation_v"]["%_eval"] = GT::percent(result.correlationV.getN(), gt);
    std::string const corr_v_filename = visualization_prefix + "-corr-v.png";
    cv::imwrite(corr_v_filename, result.corrVColor);
    json[visualization_prefix]["scores"]["correlation_v"]["visualization"]["large"] = corr_v_filename;
    json[visualization_prefix]["scores"]["correlation_v"]["visualization"]["max_val"] = result.corrVMax;
    json[visualization_prefix]["scores"]["correlation_u"]["visualization"]["normalization"] = result.corrFactor;

    // Correlation of gt_u, gt_v and diff_u, diff_v
    json[visualization_prefix]["scores"]["correlation_uv"]["value"] = result.correlationUV.getCorr();
    json[visualization_prefix]["scores"]["correlation_uv"]["%_eval"] = GT::percent(result.correlationUV.getN(), gt)/2;
    std::string const corr_uv_filename = visualization_prefix + "-corr-uv.png";
    cv::imwrite(corr_uv_filename, result.corrUVColor);
    json[visualization_prefix]["scores"]["correlation_uv"]["visualization"]["large"] = corr_uv_filename;
    json[visualization_prefix]["scores"]["correlation_uv"]["visualization"]["max_val"] = result.corrUVMax;
    json[visualization_prefix]["scores"]["correlation_uv"]["visualization"]["normalization"] = result.corrFactor;


    std::string const mee_filename = visualization_prefix + "-mee.png";
    cv::imwrite(mee_filename, result.endpointErrorVisu);
    json[visualization_prefix]["scores"]["mee"]["visualization"]["large"] = mee_filename;
    json[visualization_prefix]["scores"]["mee"]["stats"]["mean"] = result.endpointError.getMean();
    json[visualization_prefix]["scores"]["mee"]["stats"]["max"] = result.endpointError.getMax();
    for (double const quantile : quantiles) {
        json[visualization_prefix]["scores"]["mee"]["stats"][std::string("quantile-") + std::to_string(quantile)]
                = result.endpointError.getQuantile(quantile);
    }
    for (double const threshold : badpix) {
        json[visualization_prefix]["scores"]["mee"]["stats"][std::string("badpix-") + std::to_string(threshold)]
                = 100.0*(1.0-result.endpointError.getInverseQuantile(threshold));
    }

    std::string const mee_smooth_filename = visualization_prefix + "-mee-smooth.png";
    cv::imwrite(mee_smooth_filename, result.endpointErrorSmoothVisu);
    json[visualization_prefix]["scores"]["mee-smooth"]["visualization"]["large"] = mee_filename;
    json[visualization_prefix]["scores"]["mee-smooth"]["stats"]["mean"] = result.endpointErrorSmooth.getMean();
    json[visualization_prefix]["scores"]["mee-smooth"]["stats"]["max"] = result.endpointErrorSmooth.getMax();
    for (double const quantile : quantiles) {
        json[visualization_prefix]["scores"]["mee-smooth"]["stats"][std::string("quantile-") + std::to_string(quantile)]
                = result.endpointErrorSmooth.getQuantile(quantile);
    }
    for (double const threshold : badpix) {
        json[visualization_prefix]["scores"]["mee-smooth"]["stats"][std::string("badpix-") + std::to_string(threshold)]
                = 100.0*(1.0-result.endpointErrorSmooth.getInverseQuantile(threshold));
    }

    metrics_definitions["metrics"]["edge_fattening"] = {"category", "General",
            "description", "The percentage of pixels near flow edges where the algorithm result is closer to the ground truth on the other side of the edge.",
            "display_name", "EF"};
    metrics_definitions["metrics"]["edge_fattening"]["result_table_visualization"] = {
            "is_visible", true,
            "legend", "yellow = correct, red = wrong, gray = flow length of result"};

}

double FlowMetrics::naiveNearestNeighbour(
        const std::vector<cv::Point> &A,
        const std::vector<cv::Point> &B,
        const double stop_if_smaller_than) {
    double const stop_squared = stop_if_smaller_than * stop_if_smaller_than;
    double smallest_dist_sq = std::numeric_limits<double>::max();
    for (cv::Point const& a : A) {
        for (cv::Point const& b : B) {
            double const dist_sq = (a.x - b.x) * (a.x - b.x)
                    + (a.y - b.y) * (a.y - b.y);
            if (dist_sq < smallest_dist_sq) {
                smallest_dist_sq = dist_sq;
                if (smallest_dist_sq <= stop_squared) {
                    return std::sqrt(smallest_dist_sq);
                }
            }
        }
    }
    return std::sqrt(smallest_dist_sq);
}

cv::Mat FlowMetrics::get_propagated_growed_gt() {
    prepare_images();
    run_propagation(gt);
    fatteningPropagation(gt, filtered_edge_mask, propagated_gt, unc, propagated_unc);
    propagated_growed_gt = propagated_gt.clone();
    propagated_growed_unc = propagated_unc.clone();
    for (size_t ii = 0; ii < regionGrowingCounter; ++ii) {
        regionGrowing(
                    propagated_growed_gt, propagated_growed_gt,
                    propagated_growed_unc, propagated_growed_unc,
                    filtered_edge_mask);
    }
    removeInvalidFlow(gt, propagated_growed_gt);
    return propagated_growed_gt.clone();
}

cv::Mat FlowMetrics::get_propagated_growed_gt(cv::Mat const& _gt) {
    //MetricExplorer exp(_gt);
    return FlowMetrics(_gt).get_propagated_growed_gt();
}

cv::Mat FlowMetrics::get_edges() {
    prepare_images();
    run_propagation(gt);
    return filtered_edge_mask.clone();
}

cv::Mat FlowMetrics::get_edges(cv::Mat const& _gt) {
    return FlowMetrics(_gt).get_edges();
}

template<class MASK>
void FlowMetrics::findInvalidFlow(cv::Mat const& src, cv::Mat_<MASK>& mask, MASK const invalid_val) {
    int const rows = src.rows;
    int const cols = src.cols;
    if (rows != mask.rows) {
        throw std::runtime_error(std::string("src and mask row count don't match (")
                                 + std::to_string(rows) + ", " + std::to_string(mask.rows));
    }
    if (cols != mask.cols) {
        throw std::runtime_error(std::string("src and mask col count don't match (")
                                 + std::to_string(cols) + ", " + std::to_string(mask.cols));
    }
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _src = src.ptr<cv::Vec2f>(ii);
        MASK * _mask = mask.ptr(ii);
        for (int jj = 0; jj < cols; ++jj) {
            if (!GT::isValidFlow(_src[jj])) {
                _mask[jj] = invalid_val;
            }
        }
    }
}

template<class SEG, class MASK>
void FlowMetrics::filterSegments(
        cv::Mat_<SEG> const& src,
        cv::Mat_<SEG> & dst,
        cv::Mat_<MASK> & dst_mask,
        std::vector<size_t> const& sizes,
        size_t const threshold) {
    int const rows = src.rows;
    int const cols = src.cols;
    dst = src.clone();
    dst_mask = cv::Mat_<MASK>(rows, cols, 255);
    size_t const hist_size = sizes.size();
    for (int ii = 0; ii < rows; ++ii) {
        for (int jj = 0; jj < cols; ++jj) {
            SEG const seg_id = dst(ii, jj);
            size_t const seg_index = static_cast<size_t>(seg_id < 0 ? 0 : seg_id);
            if (seg_index >= hist_size) {
                size_t error_id = static_cast<size_t>(seg_id);
                size_t error_x = static_cast<size_t>(ii);
                size_t error_y = static_cast<size_t>(jj);
                size_t tmp_sum_debug = error_id + error_x + error_y;
                tmp_sum_debug *= 2;
            }
            if (seg_id >= 0 && seg_index < hist_size && sizes[seg_index] < threshold) {
                dst(ii, jj) = -1;
                dst_mask(ii, jj) = 0;
            }
            if (seg_id < 0) {
                dst_mask(ii, jj) = 0;
            }
        }
    }
}

template<class SRC, class SEG>
size_t FlowMetrics::floodFill8(cv::Mat_<SRC> const& src, cv::Mat_<SEG> & seg, int ii, int jj, size_t value) {
    size_t counter = 0;
    cv::Rect roi(0, 0, src.cols, src.rows);
    seg(ii, jj) = static_cast<int>(value);
    std::stack<cv::Point> mystack;
    mystack.push(cv::Point(jj, ii));
    while (!mystack.empty()) {
        cv::Point current = mystack.top();
        mystack.pop();
        if (src(current) > 0) {
            seg(current) = static_cast<int>(value);
            counter++;
            cv::Point candidates [8] = {
                current + cv::Point(-1, -1),
                current + cv::Point(+1, +1),
                current + cv::Point(-1, +1),
                current + cv::Point(+1, -1),
                current + cv::Point(-1,  0),
                current + cv::Point(+1,  0),
                current + cv::Point( 0, -1),
                current + cv::Point( 0, +1),
            };
            for (cv::Point const & candidate : candidates) {
                if (roi.contains(candidate) && src(candidate) > 0 && seg(candidate) < 0) {
                    seg(candidate) = static_cast<int>(value);
                    mystack.push(candidate);
                }
            }
        }
    }
    return counter;
}

template<class SRC, class SEG, class SIZE>
void FlowMetrics::getSegmentation(cv::Mat_<SRC> const& src, cv::Mat_<SEG>& segmentation, std::vector<SIZE> & segment_sizes) {
    segment_sizes.clear();
    int const rows = src.rows;
    int const cols = src.cols;
    segmentation = cv::Mat_<SEG>(rows, cols, -1);
    size_t segment_index = 0;
    for (int ii = 0; ii < rows; ++ii) {
        for (int jj = 0; jj < cols; ++jj) {
            if (src(ii, jj) > 0 && segmentation(ii, jj) < 0) {
                size_t const segment_size = floodFill8(src, segmentation, ii, jj, segment_index);
                segment_index++;
                segment_sizes.push_back(segment_size);
            }
        }
    }
}

void FlowMetrics::init(cv::Mat const& _gt, double const uncertainty) {
    cv::Mat _unc(_gt.rows, _gt.cols, CV_32FC2, cv::Vec2f(static_cast<float>(uncertainty), static_cast<float>(uncertainty)));
    init(_gt, _unc);
}

void FlowMetrics::bilateralFlowFilter(cv::Mat & src, cv::Mat & dst, int d, double sigmaColor, double sigmaSpace, int borderType) {
    cv::Mat planes[3];
    cv::split(src, planes);
    planes[2] = cv::Mat(src.size(), CV_32F, cv::Scalar(0));
    cv::Mat tmp;
    cv::merge(planes, 3, tmp);
    cv::bilateralFilter(tmp, dst, d, sigmaColor, sigmaSpace, borderType);
    cv::split(dst, planes);
    cv::merge(planes, 2, dst);
}

void FlowMetrics::init(cv::Mat const& _gt, cv::Mat const& _unc) {
    makeNeighbours(nine_neighbours, 1);
    makeNeighbours(twentyfive_neighbours, 2);
    makeNeighbours(fourtynine_neighbours, 3);
    // Check if the object was already initialized with the exact same matrizes, in this cases skip.
    if (gt.size() == _gt.size() && unc.size() == _unc.size()) {
        if (cv::norm(gt - _gt) < 1 && cv::norm(unc - _unc) < 1) {
            return;
        }
    }
    gt = _gt.clone();
    filled_gt = fillFlowHoles(gt);
    bilateralFlowFilter(filled_gt, filled_gt, 10, .07, 10);
    filled_gt_visu = GT::colorFlow(filled_gt, static_cast<float>(color_scale));

    unc = _unc.clone();
    filled_unc = fill_unc(unc);
    grad_mag = flowGradientMagnitudes(filled_gt);
    normalize_by_flow_length(filled_gt, grad_mag);

    grad_mag_visu = ColorScales::colorViridisAutoscale(grad_mag);

    ScharrVec2f(filled_gt, grad_x, grad_y);

    double min = 0, max = 0;
    cv::minMaxLoc(grad_mag, &min, &max);
    //std::cout << "min grad mag: " << min << std::endl << "max grad mag: " << max << std::endl;
    grad_mag_disp = grad_mag.clone();
    grad_mag_disp /= max;
    segmentation = grad_mag.clone();

    int const rows = gt.rows;
    int const cols = gt.cols;

    u = cv::Mat(rows, cols, CV_32F);
    grad_u_x = cv::Mat(rows, cols, CV_32F);
    grad_u_y = cv::Mat(rows, cols, CV_32F);
    v = cv::Mat(rows, cols, CV_32F);
    grad_v_x = cv::Mat(rows, cols, CV_32F);
    grad_v_y = cv::Mat(rows, cols, CV_32F);

    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _src = filled_gt.ptr<cv::Vec2f>(ii);
        float * _u = u.ptr<float>(ii);
        float * _v = v.ptr<float>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            _u[jj] = _src[jj][0];
            _v[jj] = _src[jj][1];
        }
    }
    int const ddepth = CV_32F;
    double const scale = 1.0;
    double const delta = 0;
    cv::Scharr(u, grad_u_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT);
    cv::Scharr(u, grad_u_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);

    cv::Scharr(v, grad_v_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT);
    cv::Scharr(v, grad_v_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);
}

bool FlowMetrics::isValidIndex(cv::Point const p, cv::Mat const& mat) {
    return p.x >= 0 && p.y >= 0 && p.x < mat.cols && p.y < mat.rows;
}

template<class MASK, class DIST>
void FlowMetrics::distanceMeasurement(cv::Mat_<MASK> const& edges,
                                      cv::Mat_<DIST> & distances,
                                      bool cross_edges,
                                      int const ii,
                                      int const jj,
                                      int const propagation_window_size) {
    std::queue<cv::Point> myqueue;
    DIST const maxval = std::numeric_limits<DIST>::max();
    distances = cv::Mat_<uint8_t>(2*propagation_window_size + 1, 2*propagation_window_size + 1, maxval);
    distances(propagation_window_size, propagation_window_size) = 0;
    cv::Point offset(jj - propagation_window_size, ii - propagation_window_size);
    myqueue.push(cv::Point(jj, ii));
    while(!myqueue.empty()) {
        cv::Point current = myqueue.front();
        myqueue.pop();
        cv::Point candidates [4] = {
            current + cv::Point( 0, -1),
            current + cv::Point( 0, +1),
            current + cv::Point(-1,  0),
            current + cv::Point(+1,  0),
        };
        int const current_dist = distances(current - offset);
        if (maxval == current_dist || !std::isfinite(current_dist)) {
            continue;
        }
        for (cv::Point const& candidate : candidates) {
            cv::Point const relative = candidate - offset;
            if (isValidIndex(candidate, edges) && isValidIndex(relative, distances) && distances(relative) == maxval && (cross_edges || edges(candidate) == 0)) {
                distances(relative) = std::min(int(distances(relative)), current_dist + 1);
                myqueue.push(candidate);
            }
        }
    }
}

template<class MASK>
bool FlowMetrics::localFatteningPropagation(
        cv::Mat const& gt,
        cv::Mat_<MASK> const& edges,
        cv::Mat & dst,
        cv::Mat const& unc,
        cv::Mat& unc_dst,
        int const ii, int const jj,
        cv::Mat_<cv::Vec3b>* colors) {
    int const rows = gt.rows;
    int const cols = gt.cols;

    cv::Point current(jj, ii);
    cv::Point candidates [4] = {
        current + cv::Point(-1,  0),
        current + cv::Point(+1,  0),
        current + cv::Point( 0, -1),
        current + cv::Point( 0, +1),
    };
    bool near_edge = false;
    for (cv::Point const& candidate : candidates) {
        if (isValidIndex(candidate, gt) && edges(candidate) > 0) {
            near_edge = true;
        }
    }
    if (!near_edge) {
        return false;
    }

    cv::Mat_<uint8_t> indirect_distance(2*propagation_window_size + 1, 2*propagation_window_size + 1, 255);
    indirect_distance(propagation_window_size, propagation_window_size) = 0;
    distanceMeasurement(edges, indirect_distance, false, ii, jj, propagation_window_size);

    cv::Vec2f sum(0, 0), max_unc(0, 0);
    size_t valid_counter = 0, invalid_counter = 0;
    uint8_t const maxval = std::numeric_limits<uint8_t>::max();
    for (int dii = -propagation_window_size; dii <= propagation_window_size; ++dii) {
        for (int djj = -propagation_window_size; djj <= propagation_window_size; ++djj) {
            cv::Point p(jj + djj, ii + dii);
            if (isValidIndex(p, gt)) {
                cv::Vec2f const flow = gt.at<cv::Vec2f>(p);
                cv::Point const relative = p - cv::Point(jj, ii) + cv::Point(propagation_window_size, propagation_window_size);
                auto const _indirect_dist = indirect_distance(relative);
                auto const _direct_dist = direct_distance(relative);
                if (
                        edges(p) <= 0
                        && _indirect_dist > _direct_dist
                        && maxval == _indirect_dist) {
                    cv::Vec2f const _unc = unc.at<cv::Vec2f>(p);
                    if (GT::isValidFlow(flow) && GT::isValidFlow(_unc)) {
                        sum += flow;
                        max_unc = absmax(max_unc, _unc);
                        if (colors && rows == colors->rows && cols == colors->cols) {
                            colors->operator ()(p) = cv::Vec3b(0, 0, 255);
                        }
                        valid_counter++;
                    }
                    else {
                        invalid_counter++;
                    }
                }
            }
        }
    }
    if (0 == valid_counter || invalid_counter > valid_counter || valid_counter < 3) {
        return false;
    }
    dst.at<cv::Vec2f>(ii, jj) = sum / double(valid_counter);
    unc_dst.at<cv::Vec2f>(ii, jj) = max_unc;
    return true;
}

void FlowMetrics::removeInvalidFlow(cv::Mat const& orig_gt, cv::Mat& propagated) {
    int const cols = orig_gt.cols;
    int const rows = orig_gt.rows;
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _orig = orig_gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f * _prop = propagated.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            if (!GT::isValidFlow(_orig[jj])) {
                _prop[jj] = GT::invalidFlow();
            }
        }
    }
}


bool FlowMetrics::is_distinguishable(
        double const a_val, double const a_unc,
        double const b_val, double const b_unc,
        double const threshold) {
    double const diff = std::abs(a_val - b_val);
    double const unc_sum = std::abs(a_unc) + std::abs(b_unc);
    return diff / unc_sum > threshold;
}

bool FlowMetrics::is_distinguishable(
        cv::Vec2f const a_flow, cv::Vec2f const a_unc,
        cv::Vec2f const b_flow, cv::Vec2f const b_unc,
        double const threshold) {
    return
            is_distinguishable(a_flow[0], a_unc[0], b_flow[0], b_unc[0], threshold)
            || is_distinguishable(a_flow[1], a_unc[1], b_flow[1], b_unc[1], threshold);
}

/**
     * @brief remove_similar_bg_fg Removes parts of the propagated ground truth used for measuring edge fattening
     * where the ground truth and the propagated ground truth are so similar that evaluation makes no sense
     * given the ground truth uncertainty.
     * @param gt Original ground truth
     * @param unc Original uncertainties
     * @param prop_gt Propagated ground truth
     * @param prop_unc Propagated uncertainties
     * @param threshold
     */
void FlowMetrics::remove_similar_bg_fg(
        cv::Mat const& gt, cv::Mat const& unc,
        cv::Mat& prop_gt, cv::Mat& prop_unc,
        double const threshold) {
    int const rows = gt.rows;
    int const cols = gt.cols;
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _gt = gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _unc = unc.ptr<cv::Vec2f>(ii);
        cv::Vec2f * _prop_gt = prop_gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f * _prop_unc = prop_unc.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            if (!GT::isValidFlow(_prop_gt[jj]) || !GT::isValidFlow(_prop_unc[jj])
                    || !GT::isValidFlow(_gt[jj]) || !GT::isValidFlow(_unc[jj])
                    || !is_distinguishable(_gt[jj], _unc[jj], _prop_gt[jj], _prop_unc[jj], threshold)) {
                _prop_gt[jj] = GT::invalidFlow();
                _prop_unc[jj] = GT::invalidFlow();
            }
        }
    }
}

template<class MASK>
void FlowMetrics::regionGrowing(
        cv::Mat const& orig_src,
        cv::Mat& dst,
        cv::Mat const& orig_unc,
        cv::Mat& dst_unc,
        cv::Mat_<MASK> const& edges) {
    int const rows = orig_src.rows;
    int const cols = orig_src.cols;
    cv::Mat src = orig_src.clone();
    dst = src.clone();
    cv::Mat unc = orig_unc.clone();
    dst_unc = unc.clone();
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _src = src.ptr<cv::Vec2f>(ii);
        cv::Vec2f * _dst = dst.ptr<cv::Vec2f>(ii);
        cv::Vec2f * _dst_unc = dst_unc.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            cv::Point const current(jj, ii);
            if (!GT::isValidFlow(_src[jj]) && edges(current) <= 0) {
                cv::Point const candidates [8] = {
                    current + cv::Point(-1, -1),
                    current + cv::Point(+1, +1),
                    current + cv::Point(-1, +1),
                    current + cv::Point(+1, -1),
                    current + cv::Point(-1,  0),
                    current + cv::Point(+1,  0),
                    current + cv::Point( 0, -1),
                    current + cv::Point( 0, +1),
                };
                cv::Vec2f sum(0,0), max_unc(0,0);
                size_t counter = 0;
                for (cv::Point const& candidate : candidates) {
                    if (isValidIndex(candidate, src)) {
                        cv::Vec2f const candidate_val = src.at<cv::Vec2f>(candidate);
                        cv::Vec2f const candidate_unc = unc.at<cv::Vec2f>(candidate);
                        if (edges(candidate) <= 0 && GT::isValidFlow(candidate_val) && GT::isValidFlow(candidate_unc)) {
                            sum += candidate_val;
                            max_unc = absmax(max_unc, candidate_unc);
                            counter++;
                        }
                    }
                }
                if (counter > 0) {
                    _dst[jj] = sum / (double)counter;
                    _dst_unc[jj] = max_unc;
                }
            }
        }
    }
}

template<class MASK>
void FlowMetrics::fatteningPropagation(cv::Mat const& gt, cv::Mat_<MASK> const& edges, cv::Mat & dst, cv::Mat const& unc, cv::Mat& unc_dst) {
    ParallelTime time;
    int const rows = gt.rows;
    int const cols = gt.cols;
    fattening_success = cv::Mat_<uint8_t>(rows, cols, uint8_t(0));
    dst = cv::Mat(rows, cols, CV_32FC2, GT::invalidFlow());
    unc_dst = cv::Mat(rows, cols, CV_32FC2, GT::invalidFlow());
    distanceMeasurement(edges, direct_distance, true, edges.rows/2, edges.cols/2, propagation_window_size);
    for (int ii = 0; ii< rows; ++ii) {
        for (int jj = 0; jj < cols; ++jj) {
            if (edges(ii, jj) == 0) {
                fattening_success(ii, jj) = localFatteningPropagation(gt, edges, dst, unc, unc_dst, ii, jj) ? 255 : 0;
            }
        }
    }
    //std::cout << "fatteningPropagation: " << time.print() << std::endl;
}

void FlowMetrics::run_propagation(cv::Mat const& _gt) {
    init(_gt);
    prepare_images();
    fatteningPropagation(gt, filtered_edge_mask, propagated_gt, unc, propagated_unc);
}

void FlowMetrics::run(cv::Mat const& _gt, double const scalar_unc) {
    cv::Mat _unc(_gt.rows, _gt.cols, CV_32FC2, cv::Vec2f(scalar_unc, scalar_unc));
    run(_gt, _unc);
}

void FlowMetrics::run(cv::Mat const& _gt, cv::Mat const& _unc) {
    init(_gt, _unc);

    cv::namedWindow("magnitudes");
    cv::imshow("magnitudes", grad_mag_disp);

    cv::setMouseCallback("magnitudes", CallBackFunc, this);
    while(true) {
        showImg();
        int key = cv::waitKey(0);
        switch (key % 255) {
        case 'q':
            grad_mag_threshold *= 1.1;
            needs_recalc = true;
            std::cout << "new threshold: " << grad_mag_threshold << std::endl;
            break;
        case 'a':
            grad_mag_threshold /= 1.1;
            needs_recalc = true;
            std::cout << "new threshold: " << grad_mag_threshold << std::endl;
            break;
        case '1':
            show = Showing::Gradients;
            break;
        case 'w':
            segment_size_threshold++;
            needs_recalc = true;
            std::cout << "segment size threshold: " << segment_size_threshold << std::endl;
            break;
        case 's':
            if (segment_size_threshold > 0) {
                segment_size_threshold--;
                needs_recalc = true;
            }
            std::cout << "segment size threshold: " << segment_size_threshold << std::endl;
            break;
        case '2':
            show = Showing::Filtered;
            break;

        case 'y':
            if (adaptive_block_size > 3) {
                adaptive_block_size -= 2;
                needs_recalc = true;
            }
            std::cout << "adaptive block size: " << adaptive_block_size << std::endl;
            break;
        case 'x':
            adaptive_block_size += 2;
            needs_recalc = true;
            std::cout << "adaptive block size: " << adaptive_block_size << std::endl;
            break;
        case 'c':
            adaptive_delta -= .1;
            needs_recalc = true;
            std::cout << "adaptive delta: " << adaptive_delta << std::endl;
            break;
        case 'v':
            adaptive_delta += .1;
            needs_recalc = true;
            std::cout << "adaptive delta: " << adaptive_delta << std::endl;
            break;

        case '3':
            show = Showing::FilteredAndInvalid;
            std::cout << "Showing FilteredAndInvalid" << std::endl;
            break;
        case '4':
            show = Showing::FatteningSuccess;
            std::cout << "Showing FatteningSuccess" << std::endl;
            break;
        case '5':
            show = Showing::OriginalGT;
            std::cout << "Showing OriginalGT" << std::endl;
            break;
        case 't':
            color_scale *= 1.1;
            std::cout << "New color_scale: " << color_scale << std::endl;
            needs_recalc = true;
            break;
        case 'g':
            color_scale /= 1.1;
            std::cout << "New color_scale: " << color_scale << std::endl;
            needs_recalc = true;
            break;
        case '6':
            show = Showing::PropagatedGT;
            std::cout << "Showing PropagatedGT" << std::endl;
            break;
        case '7':
            show = Showing::PropagatedGrowedGT;
            std::cout << "Showing PropagatedGrowedGT" << std::endl;
            break;
        case 'u':
            regionGrowingCounter++;
            needs_recalc = true;
            std::cout << "New regionGrowingCounter: " << regionGrowingCounter << std::endl;
            break;
        case 'j':
            if (regionGrowingCounter > 0) {
                regionGrowingCounter--;
                needs_recalc = true;
            }
            std::cout << "New regionGrowingCounter: " << regionGrowingCounter << std::endl;
            break;
        case '8':
            show = Showing::PropagatedGrowedTrimmedGT;
            std::cout << "Showing PropagatedGrowedTrimmedGT" << std::endl;
            break;
        case '9':
            show = Showing::PropagatedGrowedTrimmedDistinguishableGT;
            std::cout << "Showing PropagatedGrowedTrimmedDistinguishableGT" << std::endl;
            break;
        case 'o':
            similarity_threshold *= 1.1;
            needs_recalc = true;
            std::cout << "New similarity_threshold: " << similarity_threshold << std::endl;
            break;
        case 'l':
            similarity_threshold /= 1.1;
            needs_recalc = true;
            std::cout << "New similarity_threshold: " << similarity_threshold << std::endl;
            break;
        case '0':
            show = Showing::AlgoResult;
            std::cout << "Showing AlgoResult" << std::endl;
            break;
        case 'p':
            show = Showing::AlgoFattening;
            std::cout << "Showing AlgoFattening" << std::endl;
            break;
        }


    }
}

void FlowMetrics::normalize_by_flow_length(cv::Mat const& flow, cv::Mat& srcdst, double epsilon) {
    int const cols = flow.cols;
    int const rows = flow.rows;

    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _flow = flow.ptr<cv::Vec2f>(ii);
        float * _srcdst = srcdst.ptr<float>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            if (GT::isValidFlow(_flow[jj])) {
                double const flow_length = std::sqrt(_flow[jj][0]*_flow[jj][0] + _flow[jj][1]*_flow[jj][1]);
                if (std::isfinite(flow_length)) {
                    double const divisor = std::max(epsilon, flow_length);
                    _srcdst[jj] /= divisor;
                }
            }
        }
    }
}

void FlowMetrics::computeFineStructureNeighbours() {
    fine_structure_neighbours = cv::Mat_<uint8_t>(fine_structures.rows, fine_structures.cols, uint8_t(0));
    for (int ii = 0; ii < fine_structures.rows; ++ii) {
        for (int jj = 0; jj < fine_structures.cols; ++jj) {
            cv::Point const point(jj, ii);
            if (fine_structures(point) != 0 || filtered_edge_mask(point) != 0) {
                continue;
            }
            if (l2distance2fine_structures(point) <= fine_structure_grow_distance) {
                fine_structure_neighbours(point) = 255;
            }
        }
    }
}

template<class DATA, class MASK>
cv::Mat_<DATA> setTo(
        cv::Mat_<DATA> const& src,
        cv::Mat_<MASK> const& mask,
        DATA const value,
        bool inverted = false) {
    cv::Mat_<DATA> result = src.clone();
    for (int ii = 0; ii < result.rows; ++ii) {
        for (int jj = 0; jj < result.cols; ++jj) {
            cv::Point const p(jj, ii);
            if ((mask(p) != 0) ^ inverted) {
                result(p) = value;
            }
        }
    }
    return result;
}

void FlowMetrics::prepare_images() {
    ParallelTime total;
    if (!needs_recalc) {
        return;
    }
    needs_recalc = false;
    //cv::threshold(grad_mag, grad_mag_disp, grad_mag_threshold, 1.0, cv::THRESH_BINARY);
    double min = 0, max = 0;
    cv::minMaxLoc(grad_mag, &min, &max);
    cv::Mat grad_mag_scaled;
    double const alpha = 255 / (max - min);
    double const beta = -min / alpha;
    grad_mag.convertTo(grad_mag_scaled, CV_8UC1, alpha, beta);

    //    void cv::calcHist 	( 	const Mat *  	images,
    //            int  	nimages,
    //            const int *  	channels,
    //            InputArray  	mask,
    //            OutputArray  	hist,
    //            int  	dims,
    //            const int *  	histSize,
    //            const float **  	ranges,
    //            bool  	uniform = true,
    //            bool  	accumulate = false
    //        )
    if (save_all_images) {
        plotHist(grad_mag, "gradient-strength");
    }
    {
        cv::Mat_<float> meanfloat,diff_float;
        meanfloat = grad_mag.clone();
        GaussianBlur(grad_mag, meanfloat,
                     cv::Size(adaptive_block_size, adaptive_block_size),
                     0.0, 0.0);
                     //cv::BORDER_REPLICATE|cv::BORDER_ISOLATED);
        diff_float = grad_mag - meanfloat;
        adaptive_threshold_visu = ColorScales::colorViridisAutoscale(diff_float);
    }
    cv::adaptiveThreshold(grad_mag_scaled,
                          unfiltered_edges,
                          255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY,
                          adaptive_block_size,
                          adaptive_delta
                          );


    cv::Mat_<uint8_t> inverted_unfiltered_edge_mask = cv::Scalar(255) - unfiltered_edges;
    cv::distanceTransform(inverted_unfiltered_edge_mask, l2distance2unfiltered_edges, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    cv::Mat _unfiltered_filtered_edge_mask_grown;
    cv::threshold(l2distance2unfiltered_edges, _unfiltered_filtered_edge_mask_grown, double(unfilteredEdgeGrowingCounter), 255, cv::THRESH_BINARY);
    _unfiltered_filtered_edge_mask_grown.convertTo(unfiltered_edges_grown, unfiltered_edges_grown.type());
    unfiltered_edges_grown = cv::Scalar(255) - unfiltered_edges_grown;

    unfiltered_edges.convertTo(grad_mag_disp, CV_32FC1);
    getSegmentation(grad_mag_disp, segmentation, segment_sizes);
    filterSegments(segmentation, filtered_segmentation, filtered_edge_mask, segment_sizes, segment_size_threshold);
    closeSegmentGaps(filtered_segmentation, filtered_segmentation, filtered_edge_mask);
    findInvalidFlow(gt, filtered_edge_mask);
    filterSegments(filtered_edge_mask, filtered_edge_mask, segment_size_threshold);
    filtered_edge_mask = make4neighbours8neighbours(filtered_edge_mask);
    fatteningPropagation(gt, filtered_edge_mask, propagated_gt, unc, propagated_unc);
    filtered_edge_mask_grown = filtered_edge_mask.clone();

    findFineStructureGlobal(filtered_edge_mask, fine_structures, false);
    cv::Mat_<uint8_t> inverted_fine_structures = cv::Scalar(255) - fine_structures;
    cv::distanceTransform(inverted_fine_structures, l2distance2fine_structures, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    computeFineStructureNeighbours();

    filterSegments(fine_structure_neighbours, fine_structure_neighbours, segment_size_threshold);


    cv::Mat_<uint8_t> inverted_edge_mask = cv::Scalar(255) - filtered_edge_mask;
    cv::distanceTransform(inverted_edge_mask, l2distance2edges, cv::DIST_L2, cv::DIST_MASK_PRECISE);


    l2distance2edges_visu = ColorScales::colorViridisAutoscale(l2distance2edges);

    cv::Mat _filtered_edge_mask_grown;
    cv::threshold(l2distance2edges, _filtered_edge_mask_grown, double(edgeGrowingCounter), 255, cv::THRESH_BINARY);
    _filtered_edge_mask_grown.convertTo(filtered_edge_mask_grown, filtered_edge_mask_grown.type());
    filtered_edge_mask_grown = cv::Scalar(255) - filtered_edge_mask_grown;

    colored_gt = GT::colorFlow(gt, color_scale);
    colored_propagated_gt = GT::colorFlow(propagated_gt, color_scale);

    propagated_growed_gt = propagated_gt.clone();
    propagated_growed_unc = propagated_unc.clone();
    for (size_t ii = 0; ii < regionGrowingCounter; ++ii) {
        regionGrowing(
                    propagated_growed_gt, propagated_growed_gt,
                    propagated_growed_unc, propagated_growed_unc,
                    filtered_edge_mask);
    }
    colored_propagated_growed_gt = GT::colorFlow(propagated_growed_gt, color_scale);

    propagated_growed_trimmed_gt = propagated_growed_gt.clone();
    removeInvalidFlow(gt, propagated_growed_trimmed_gt);

    colored_propagated_growed_trimmed_gt = GT::colorFlow(propagated_growed_trimmed_gt, color_scale);

    propagated_growed_trimmed_distinguishable_gt = propagated_growed_trimmed_gt.clone();
    propagated_growed_trimmed_distinguishable_unc = propagated_growed_unc.clone();
    remove_similar_bg_fg(gt, unc,
                         propagated_growed_trimmed_distinguishable_gt, propagated_growed_trimmed_distinguishable_unc,
                         similarity_threshold);

    colored_propagated_growed_trimmed_distinguishable_gt
            = GT::colorFlow(propagated_growed_trimmed_distinguishable_gt, color_scale);

    //res = evaluateFattening(algo);

    colored_algo = GT::colorFlow(algo, color_scale);
    gt_invalid_mask = cv::Mat_<uint8_t>(fine_structures.rows, fine_structures.cols, uint8_t(0));
    fine_structure_mask_visu = cv::Mat_<cv::Vec3b>(fine_structures.rows, fine_structures.cols, cv::Vec3b(255,255,255));
    for (int ii = 0; ii < fine_structures.rows; ++ii) {
        for (int jj = 0; jj < fine_structures.cols; ++jj) {
            uint8_t red = 0, green = 0, blue = 0;
            cv::Point const point(jj, ii);
            if (!GT::isValidFlow(gt(point))) {
                gt_invalid_mask(point) = 255;
            }
            if (filtered_edge_mask(point) != 0) {
                fine_structure_mask_visu(point) = cv::Vec3b(0,0,0);
                continue;
            }
            bool eval = false;
            if (fine_structures(point) != 0) {
                eval = true;
                fine_structure_mask_visu(point) = ColorsRD::green();
            }
            if (fine_structure_neighbours(point) != 0) {
                eval = true;
                fine_structure_mask_visu(point) = ColorsRD::purple();
            }
            if (eval && !GT::isValidFlow(propagated_growed_trimmed_distinguishable_gt.at<cv::Vec2f>(point))) {
                fine_structure_mask_visu(point) = ColorsRD::red();
            }
        }
    }

    setBorder(filtered_edge_mask, ignore_border, uint8_t(255));
    setBorder(filtered_edge_mask_grown, ignore_border, uint8_t(255));

    smoothing_mask = filtered_edge_mask_grown.clone();
    smoothing_mask = setTo(smoothing_mask, unfiltered_edges_grown, uint8_t(255));

    colored_propagated_growed_gt_with_edges = setTo(colored_propagated_growed_gt, filtered_edge_mask, cv::Vec3b(0,0,255));
    setBorder(colored_propagated_growed_gt_with_edges, ignore_border, cv::Vec3b(0,0,0));

    colored_propagated_growed_gt_edges = setTo(colored_propagated_growed_gt, filtered_edge_mask, cv::Vec3b(255,0,0));
    setBorder(colored_propagated_growed_gt_edges, ignore_border, cv::Vec3b(0,0,0));

    filtered_edge_mask_no_border = filtered_edge_mask.clone();
    setBorder(filtered_edge_mask_no_border, ignore_border, uint8_t(0));

    //std::cout << "time: " << total.print() << std::endl;
}


template<class SEG>
cv::Mat_<cv::Vec3b> FlowMetrics::colorSegmentation(const cv::Mat_<SEG> &segmentation) {
    cv::Mat_<cv::Vec3b> result(segmentation.rows, segmentation.cols);
    throw(0);
    return result;
}



void FlowMetrics::showImg() {
    prepare_images();
    std::cout << "gt stats: " << std::endl << GT::printStats(gt) << std::endl;
    switch (show) {
    case Showing::Gradients:
        cv::imshow("magnitudes", grad_mag_disp);
        break;
    case Showing::Filtered:
        cv::imshow("magnitudes", filtered_edge_mask);
        break;
    case Showing::FilteredAndInvalid:
        cv::imshow("magnitudes", filtered_edge_mask);
        break;
    case Showing::FatteningSuccess:
        cv::imshow("magnitudes", fattening_success);
        break;
    case Showing::OriginalGT:
        cv::imshow("magnitudes", colored_gt);
        break;
    case Showing::PropagatedGT:
        cv::imshow("magnitudes", colored_propagated_gt);
        break;
    case Showing::PropagatedGrowedGT:
        cv::imshow("magnitudes", colored_propagated_growed_gt);
        std::cout << "growed flow stats: " << std::endl << GT::printStats(propagated_growed_gt) << std::endl;
        break;
    case Showing::PropagatedGrowedTrimmedGT:
        cv::imshow("magnitudes", colored_propagated_growed_trimmed_gt);
        std::cout << "growed flow stats: " << std::endl << GT::printStats(propagated_growed_trimmed_gt) << std::endl;
        break;
    case Showing::PropagatedGrowedTrimmedDistinguishableGT:
        cv::imshow("magnitudes", colored_propagated_growed_trimmed_distinguishable_gt);
        std::cout << "growed flow stats: " << std::endl << GT::printStats(propagated_growed_trimmed_distinguishable_gt) << std::endl;
        break;
    case Showing::AlgoResult:
        cv::imshow("magnitudes", colored_algo);
        break;
    case Showing::AlgoFattening:
        cv::imshow("magnitudes", res.coloredResult);
        std::cout << res.printStats() << std::endl;
        break;
    }

}

void bumpinessPostprocessing(cv::Mat_<cv::Vec3b> & visu, cv::Mat_<float> data) {
    for (int ii = 0; ii < data.rows; ++ii) {
        for (int jj = 0; jj < data.cols; ++jj) {
            cv::Point const p(jj, ii);
            if (data(p) <= 0 && visu(p) != cv::Vec3b(255, 255, 255)) {
                visu(p) = cv::Vec3b(0,0,0);
            }
        }
    }
}

FatteningResult FlowMetrics::evaluateFattening(cv::Mat const& submission) {
    prepare_images();

    algo = submission.clone();

    FatteningResult res;

    res.corrURaw = cv::Mat_<float>(gt.size(), 0.0);
    res.corrVRaw = cv::Mat_<float>(gt.size(), 0.0);
    res.corrUVRaw = cv::Mat_<float>(gt.size(), 0.0);
    res.smoothingData = cv::Mat_<float>(gt.size(), 0.0);
    res.bumpinessData = cv::Mat_<float>(gt.size(), 0.0);
    res.endpointErrorData = cv::Mat_<float>(gt.size(), 0.0);
    res.endpointErrorSmoothData = cv::Mat_<float>(gt.size(), 0.0);

    cv::Mat_<uint8_t> inverse_smoothing_mask = cv::Scalar(255) - smoothing_mask;

    cv::Point max_gt_gradient_masked_point;
    double max_gt_gradient_masked = 0;
    cv::minMaxLoc(grad_mag, nullptr, &max_gt_gradient_masked, nullptr, &max_gt_gradient_masked_point, inverse_smoothing_mask);
    res.smoothingVisuScale = res.bumpinessVisuScale = 0;

    cv::Point max_gt_gradient_point;
    double max_gt_gradient = 0;
    cv::minMaxLoc(grad_mag, nullptr, &max_gt_gradient, nullptr, &max_gt_gradient_point);

    cv::Scalar _mean_gradient, _stddev_gradient;
    cv::meanStdDev(grad_mag, _mean_gradient, _stddev_gradient, inverse_smoothing_mask);
    double const mean_gradient = _mean_gradient[0];
    double const stddev_gradient = _stddev_gradient[0];

    cv::Mat_<cv::Vec2f> const filled_submission = fillFlowHoles(submission);

    cv::Mat_<float> const submission_grad_mag = flowGradientMagnitudes(submission);


    prepare_images();
    if (algo.cols != gt.cols || algo.rows != gt.rows) {
        throw std::runtime_error(std::string("gt and algo sizes differ: ") +
                                 GT::matsize(gt) + " vs. " + GT::matsize(algo));
    }

    int const cols = algo.cols;
    int const rows = algo.rows;


    res.coloredResult = GT::grayFlow(algo, GT::maxFlowLength(gt));
    res.fineFatteningVisu = res.coloredResult.clone();
    res.fineThinningVisu = res.coloredResult.clone();

    res.differenceFlow = submission - gt;
    res.differenceFlowColored = GT::colorFlow(res.differenceFlow, 1);

    cv::Mat_<float> differenceFlowComponents[2];
    cv::split(res.differenceFlow, differenceFlowComponents);

    cv::Mat_<float> algoComponents[2];
    cv::split(algo, algoComponents);

    QuantileStats<float> bumpiness, smoothing;


    for (int ii = ignore_border; ii+ignore_border < rows; ++ii) {
        cv::Vec2f const * const _gt       = gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _unc      = unc.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _prop_gt  = propagated_growed_trimmed_distinguishable_gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _prop_unc = propagated_growed_trimmed_distinguishable_unc.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _algo     = algo.ptr<cv::Vec2f>(ii);
        cv::Vec3b * _color = res.coloredResult.ptr<cv::Vec3b>(ii);
        for (int jj = ignore_border; jj+ignore_border < cols; ++jj) {
            if (!GT::isValidFlow(_unc[jj])) {
                continue;
            }
            if (!GT::isValidFlow(_gt[jj])) {
                continue;
            }
            res.gtLength.push(GT::flowLength(_gt[jj]));
            cv::Point const point(jj, ii);

            if (!GT::isValidFlow(_algo[jj])) {
                res.invalidPixels.push(true);
                if (GT::isValidFlow(_prop_gt[jj]) && GT::isValidFlow(_prop_unc[jj])) {
                    _color[jj] = bad_color;
                    res.edgeFattening.push(true);
                    if (fine_structures(point) != 0) {
                        res.fineThinningVisu(point) = bad_color;
                        res.fineThinning.push(true);
                    }
                    if (fine_structure_neighbours(point) != 0) {
                        res.fineFatteningVisu(point) = bad_color;
                        res.fineFattening.push(true);
                    }
                }
                continue;
            }
            double const local_endpoint_error = GT::dist(_gt[jj], _algo[jj]);
            res.endpointError.push(local_endpoint_error);
            res.endpointErrorData(point) = local_endpoint_error;
            res.invalidPixels.push(false);

            if (smoothing_mask(point) == 0) {
                res.endpointErrorSmooth.push(local_endpoint_error);
                res.endpointErrorSmoothData(point) = local_endpoint_error;

                float const gradient_diff = grad_mag(point) - submission_grad_mag(point);

                bool const has_bumpiness = submission_grad_mag(point) > grad_mag(point);
                float const local_bumpiness = std::max(0.0f, -gradient_diff);
                res.bumpinessCount.push(has_bumpiness);
                res.bumpinessData(point) = local_bumpiness;
                res.bumpiness.push(local_bumpiness);
                res.bumpinessVisuScale = std::max<double>(res.bumpinessVisuScale, local_bumpiness);
                bumpiness.push(local_bumpiness);

                bool const has_smoothing = grad_mag(point) > submission_grad_mag(point);
                float const local_smoothing = std::max(0.0f, gradient_diff);
                res.smoothingCount.push(has_smoothing);
                res.smoothingData(point) = local_smoothing;
                res.smoothing.push(local_smoothing);
                res.smoothingVisuScale = std::max<double>(res.smoothingVisuScale, local_bumpiness);
                smoothing.push(local_smoothing);
            }

            cv::Vec2f const current_diff = _algo[jj] - _gt[jj];
            res.correlationU.push(_algo[jj][0], current_diff[0]);
            res.correlationV.push(_algo[jj][1], current_diff[1]);

            res.correlationUV.push(_algo[jj][0], current_diff[0]);
            res.correlationUV.push(_algo[jj][1], current_diff[1]);


            if (!GT::isValidFlow(_prop_gt[jj])) {
                continue;
            }
            if (!GT::isValidFlow(_prop_unc[jj])) {
                continue;
            }


            if (GT::dist_sq(_gt[jj], _algo[jj]) > GT::dist_sq(_prop_gt[jj], _algo[jj])) {
                res.edgeFattening.push(true);
                _color[jj] = bad_color;
                if (fine_structures(point) != 0) {
                    res.fineThinningVisu(point) = bad_color;
                    res.fineThinning.push(true);
                }
                if (fine_structure_neighbours(point) != 0) {
                    res.fineFatteningVisu(point) = bad_color;
                    res.fineFattening.push(true);
                }
            }
            else {
                _color[jj] = good_color;
                res.edgeFattening.push(false);
                if (fine_structures(point) != 0) {
                    res.fineThinningVisu(point) = good_color;
                    res.fineThinning.push(false);
                }
                if (fine_structure_neighbours(point) != 0) {
                    res.fineFatteningVisu(point) = good_color;
                    res.fineFattening.push(false);
                }
            }
        }
    }

    for (int ii = ignore_border; ii+ignore_border < rows; ++ii) {
        for (int jj = ignore_border; jj+ignore_border < cols; ++jj) {
            cv::Point const point(jj, ii);
            //res.bumpinessData(point) = bumpiness.getInverseQuantile(res.bumpinessData(point));
            //res.smoothingData(point) = smoothing.getInverseQuantile(res.smoothingData(point));
        }
    }

    //res.bumpinessVisu = GT::colorViridisAutoscale(res.bumpinessData);
    //res.smoothingVisu = GT::colorViridisAutoscale(res.smoothingData);

    res.bumpinessVisu = ColorScales::colorViridisScaled(res.bumpinessData, res.bumpiness.getQuantile(.97));
    res.smoothingVisu = ColorScales::colorViridisScaled(res.smoothingData, res.smoothing.getQuantile(.97));

    //setBorder(res.bumpinessVisu, ignore_border, cv::Vec3b(255,255,255));
    //setBorder(res.smoothingVisu, ignore_border, cv::Vec3b(255,255,255));

    res.bumpinessVisu = setTo(res.bumpinessVisu, smoothing_mask, cv::Vec3b(255,255,255));
    res.smoothingVisu = setTo(res.smoothingVisu, smoothing_mask, cv::Vec3b(255,255,255));

    bumpinessPostprocessing(res.bumpinessVisu, res.bumpinessData);
    bumpinessPostprocessing(res.smoothingVisu, res.smoothingData);


    res.corrURaw = cv::abs((algoComponents[0] - cv::Scalar(res.correlationU.getMeanX()))
            .mul(differenceFlowComponents[0] - cv::Scalar(res.correlationU.getMeanY())));

    res.corrVRaw = cv::abs((algoComponents[1] - cv::Scalar(res.correlationV.getMeanX()))
            .mul(differenceFlowComponents[1] - cv::Scalar(res.correlationV.getMeanY())));

    res.corrUVRaw = res.corrURaw + res.corrURaw;

    cv::minMaxLoc(res.corrURaw, nullptr, &res.corrUMax);
    cv::minMaxLoc(res.corrVRaw, nullptr, &res.corrVMax);
    cv::minMaxLoc(res.corrUVRaw, nullptr, &res.corrUVMax);

    res.corrMax = std::max(res.corrUMax, std::max(res.corrVMax, res.corrUVMax));

    double const sigma_factor = 5;
    cv::Scalar tmp_mean, tmp_stddev;

    cv::meanStdDev(res.corrURaw, tmp_mean, tmp_stddev);
    res.corrFactor = std::max(res.corrFactor, tmp_mean[0] + sigma_factor * tmp_stddev[0]);

    cv::meanStdDev(res.corrVRaw, tmp_mean, tmp_stddev);
    res.corrFactor = std::max(res.corrFactor, tmp_mean[0] + sigma_factor * tmp_stddev[0]);

    cv::meanStdDev(res.corrUVRaw, tmp_mean, tmp_stddev);
    res.corrFactor = std::max(res.corrFactor, tmp_mean[0] + sigma_factor * tmp_stddev[0]);

    res.corrUColor = ColorScales::colorViridisScaled(res.corrURaw, res.corrFactor);
    res.corrVColor = ColorScales::colorViridisScaled(res.corrVRaw, res.corrFactor);
    res.corrUVColor = ColorScales::colorViridisScaled(res.corrUVRaw, res.corrFactor);

    res.endpointErrorVisu = ColorScales::colorViridisScaled(res.endpointErrorData, res.gtLength.getMax());
    res.endpointErrorVisu = setTo(res.endpointErrorVisu, gt_invalid_mask, cv::Vec3b(0,0,0));

    res.endpointErrorSmoothVisu = ColorScales::colorViridisScaled(res.endpointErrorSmoothData, res.gtLength.getMax());
    res.endpointErrorSmoothVisu = setTo(res.endpointErrorSmoothVisu, filtered_edge_mask_grown, cv::Vec3b(0,0,0));

    /*
    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _gt       = gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _unc      = unc.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _prop_gt  = propagated_growed_trimmed_distinguishable_gt.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _prop_unc = propagated_growed_trimmed_distinguishable_unc.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const _algo     = algo.ptr<cv::Vec2f>(ii);
        cv::Vec3b * _color = res.coloredResult.ptr<cv::Vec3b>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            if (!GT::isValidFlow(_unc[jj])) {
                continue;
            }
            if (!GT::isValidFlow(_gt[jj])) {
                continue;
            }
            res.numEval++;
            if (!GT::isValidFlow(_algo[jj])) {
                res.numBad++;
                res.numAlgoInvalid++;
                continue;
            }

            cv::Vec2f const current_diff = _algo[jj] - _gt[jj];
            res.correlationU.push(_algo[jj][0], current_diff[0]);
            res.correlationV.push(_algo[jj][1], current_diff[1]);

            res.correlationUV.push(_algo[jj][0], current_diff[0]);
            res.correlationUV.push(_algo[jj][1], current_diff[1]);

            res.corrURaw(ii, jj) = (_algo[jj][0] - res.correlationU.getMeanX())
                    * (current_diff[0] - res.correlationU.getMeanY());

            res.corrVRaw(ii, jj) = (_algo[jj][1] - res.correlationV.getMeanX())
                    * (current_diff[1] - res.correlationU.getMeanY());

            if (!GT::isValidFlow(_prop_gt[jj])) {
                continue;
            }
            if (!GT::isValidFlow(_prop_unc[jj])) {
                continue;
            }


            if (GT::dist_sq(_gt[jj], _algo[jj]) > GT::dist_sq(_prop_gt[jj], _algo[jj])) {
                res.numBad++;
                _color[jj] = bad_color;
            }
            else {
                res.numGood++;
                _color[jj] = good_color;
            }

        }
    }
    */

    colored_algo = GT::colorFlow(algo, color_scale);
    colored_algo = setTo(colored_algo, filtered_edge_mask_no_border, cv::Vec3b(0,0,0));

    res.fineErrors = BinaryStats::merged(res.fineThinning, res.fineFattening);


    return res;
}



template<class SEG, class MASK>
void FlowMetrics::closeSegmentGaps(const cv::Mat_<SEG> & src,
                                   cv::Mat_<SEG> & dst,
                                   cv::Mat_<MASK> & mask) {
    int const rows = src.rows;
    int const cols = src.cols;
    cv::Mat_<SEG> tmp = src.clone();
    dst = src.clone();

    size_t gaps_closed = 0;

    for (int ii = 0; ii < rows; ++ii) {
        for (int jj = 0; jj < cols; ++jj) {
            cv::Point current = cv::Point(jj, ii);
            if (dst(current) >= 0 ) {
                continue;
            }
            cv::Point candidates [8] = {
                current + cv::Point(-1, -1),
                current + cv::Point(+1, +1),
                current + cv::Point(-1, +1),
                current + cv::Point(+1, -1),
                current + cv::Point(-1,  0),
                current + cv::Point(+1,  0),
                current + cv::Point( 0, -1),
                current + cv::Point( 0, +1)
            };
            for (cv::Point const & c1 : candidates) {
                if (isValidIndex(c1, tmp)
                        && tmp(c1) >= 0) {
                    for (cv::Point const & c2 : candidates) {
                        if (isValidIndex(c2, tmp)
                                && tmp(c2) >= 0
                                && tmp(c1) != tmp(c2)) {
                            dst(current) = tmp(c1);
                            gaps_closed++;
                            mask(current) = 255;
                        }
                    }
                }
            }
        }
    }

}

template<class MASK>
void FlowMetrics::filterSegments(
        const cv::Mat_<MASK> &src,
        cv::Mat_<MASK> &dst,
        const size_t threshold) {
    cv::Mat_<int> local_segmentation, local_filtered_segmentation;
    std::vector<size_t> local_segment_sizes;
    dst = src.clone();
    getSegmentation(src, local_segmentation, local_segment_sizes);
    filterSegments(local_segmentation, local_filtered_segmentation, dst, local_segment_sizes, threshold);
}

template<class MASK>
void FlowMetrics::findPlanes(const cv::Mat &flow, const cv::Mat_<MASK> &edges) {

}

template<class MASK, class BAD, class RES>
void findPlaneDebugImage(
        const cv::Mat_<MASK> &edges,
        const cv::Mat_<BAD> & bad_pixels,
        const cv::Mat_<RES> & plane_img,
        cv::Point const current
        ) {

    cv::Mat_<cv::Vec3b> result(edges.size(), cv::Vec3b(0,0,0));

    for (int ii = 0; ii < result.rows; ++ii) {
        for (int jj = 0; jj < result.cols; ++jj) {
            if (edges(ii, jj) > 0) {
                result(ii, jj) = cv::Vec3b(255,255,255);
            }
            if (plane_img(ii, jj) > 0) {
                result(ii, jj) = cv::Vec3b(0,255,0);
            }
            if (bad_pixels(ii, jj) > 0) {
                result(ii, jj) = cv::Vec3b(0,0,255);
            }
        }
    }
    result(current) = cv::Vec3b(255,0,0);

    cv::imshow("magnitudes", result);
}

template<class MASK, class BAD, class RES>
void findPlaneDebugImage2(
        const cv::Mat_<MASK> &edges,
        const cv::Mat_<BAD> & bad_pixels,
        const cv::Mat_<RES> & plane_img,
        std::vector<cv::Point> const& current_fringe,
        std::vector<cv::Point> const& next_fringe
        ) {

    cv::Mat_<cv::Vec3b> result(edges.size(), cv::Vec3b(0,0,0));

    for (int ii = 0; ii < result.rows; ++ii) {
        for (int jj = 0; jj < result.cols; ++jj) {
            if (edges(ii, jj) > 0) {
                result(ii, jj) = cv::Vec3b(255,255,255);
            }
            if (plane_img(ii, jj) > 0) {
                result(ii, jj) = cv::Vec3b(0,255,0);
            }
            if (bad_pixels(ii, jj) > 0) {
                result(ii, jj) = cv::Vec3b(0,0,255);
            }
        }
    }
    for (cv::Point const& cf : current_fringe) {
        result(cf) = cv::Vec3b(0,255,255);
    }
    for (cv::Point const& nf : next_fringe) {
        result(nf) = cv::Vec3b(255,0,0);
    }

    cv::imshow("magnitudes", result);
}

template<class MASK>
bool FlowMetrics::findPlane(const cv::Mat &flow, const cv::Mat_<MASK> &edges, const cv::Point2i initial)
{
    cv::Point candidates [9] = {
        initial + cv::Point(-1, -1),
        initial + cv::Point(+1, +1),
        initial + cv::Point(-1, +1),
        initial + cv::Point(+1, -1),
        initial + cv::Point(-1,  0),
        initial + cv::Point(+1,  0),
        initial + cv::Point( 0, -1),
        initial + cv::Point( 0, +1),
        initial
    };

    cv::Mat_<uint8_t> plane_img(GT::convertSize(flow.size), uint8_t(0));
    std::vector<cv::Point2d> plane_list;
    std::vector<cv::Point2d> src, dst;
    for (cv::Point const & candidate : candidates) {
        if (isValidIndex(candidate, flow)
                && GT::isValidFlow(flow.at<cv::Vec2f>(candidate))
                && edges(candidate) == 0) {
            src.push_back(candidate);
            dst.push_back(cv::Point2d(candidate) + cv::Point2d(flow.at<cv::Vec2f>(candidate)));
            plane_img(candidate) = 255;
        }
        else {
            return false;
        }
    }


    HomographyFlow current_flow;
    ContourFlow::FlowFitter<HomographyFlow> sub(current_flow, src, dst, -1);
    current_flow = sub.result;
    std::cout << sub.summary.FullReport() << std::endl;

    double max_err = maxError(current_flow, src, dst);
    if (max_err >= same_plane_threshold) {
        return false;
    }

    size_t counter = 0;
    cv::Rect roi(0, 0, flow.cols, flow.rows);
    std::stack<cv::Point> mystack;
    for (int dx = -1; dx <= 1; ++dx) {
        mystack.push(initial + cv::Point(dx,  2));
        mystack.push(initial + cv::Point(dx, -2));
        mystack.push(initial + cv::Point( 2, dx));
        mystack.push(initial + cv::Point(-2, dx));
    }
    mystack.push(initial + cv::Point( 2,  2));
    mystack.push(initial + cv::Point(-2, -2));
    mystack.push(initial + cv::Point(-2,  2));
    mystack.push(initial + cv::Point( 2, -2));
    cv::Mat_<uint8_t> bad_pixels(GT::convertSize(flow.size), uint8_t(0));
    size_t iteration_counter = 0;
    cv::Point current = mystack.top();
    ParallelTime t;
    while (!mystack.empty()) {
        iteration_counter++;

        current = mystack.top();
        mystack.pop();
        if (isValidIndex(current, flow)
                && GT::isValidFlow(flow.at<cv::Vec2f>(current))
                && edges(current) == 0
                && bad_pixels(current) == 0) {
            if (plane_img(current) == 0) {
                t.start();
                findPlaneDebugImage(edges, bad_pixels, plane_img, current);
                std::string const paint_time = t.print();
                cv::Point2d current_dst = cv::Point2d(current) + cv::Point2d(flow.at<cv::Vec2f>(current));
                cv::Point2d residual = current_flow.apply(current) - current_dst;
                double const error = std::sqrt(residual.dot(residual));
                if (error < same_plane_threshold) {
                    counter++;
                    plane_img(current) = 255;

                    src.push_back(current);
                    dst.push_back(current_dst);
                    sub.addPoint(current, current_dst, -1);
                    t.start();
                    sub.solve();
                    std::string const solve_time = t.print();
                    current_flow = sub.result;
                    //std::cout << sub.summary.FullReport() << std::endl;
                    max_err = maxError(current_flow, src, dst);
                    if (iteration_counter % 20 == 0) {
                        std::cout << "Max error: " << max_err << std::endl;
                        std::cout << "Paint time: " << paint_time << std::endl;
                        std::cout << "Solve time: " << solve_time << std::endl;
                    }
                    cv::Point candidates [8] = {
                        current + cv::Point(-1, -1),
                        current + cv::Point(+1, +1),
                        current + cv::Point(-1, +1),
                        current + cv::Point(+1, -1),
                        current + cv::Point(-1,  0),
                        current + cv::Point(+1,  0),
                        current + cv::Point( 0, -1),
                        current + cv::Point( 0, +1),
                    };
                    for (cv::Point const & candidate : candidates) {
                        if (isValidIndex(candidate, flow)
                                && GT::isValidFlow(flow.at<cv::Vec2f>(candidate))
                                && edges(candidate) == 0
                                && plane_img(candidate) == 0
                                && bad_pixels(candidate) == 0) {
                            mystack.push(candidate);
                        }
                    }
                }
                else {
                    bad_pixels(current) = 255;
                }
            }
        }
        else {
            bad_pixels(current) = 255;
        }
    }
    findPlaneDebugImage(edges, bad_pixels, plane_img, current);

    std::cout << "Iterations: " << iteration_counter << ", valid: " << counter << std::endl;

    max_err = maxError(current_flow, src, dst);
    std::cout << "Max error: " << max_err << std::endl;
    return true;
}

template<class MASK, class FLOW>
bool FlowMetrics::findArea(const cv::Mat &flow, const cv::Mat_<MASK> &edges, const cv::Point2i initial)
{
    ParallelTime function_time;

    ContourFlow::FlowFitter<FLOW> sub;

    int min_max_dx = 1;


    FLOW current_flow;
    // If the numer of parameters is 9 or smaller a 3x3 window is large enough
    // so that we have 2 values per parameter.

    // We need more given data points than the number of parameters.
    // We want the first fit to be rather over-determined so we
    // introduce the dof_factor and use a initial window large enough
    // s.t. we have at least dof_factor * num_params initial points.
    double const dof_factor = 3;
    for (min_max_dx = 1; min_max_dx <= 15; min_max_dx++) {
        size_t const window_width = 2 * size_t(min_max_dx) + 1;
        size_t const resulting_data = 2 * window_width * window_width;
        if (resulting_data >= dof_factor * current_flow.num_params) {
            break;
        }
    }




    cv::Mat_<uint8_t> plane_img(GT::convertSize(flow.size), uint8_t(0));
    cv::Mat_<uint8_t> visited_img(GT::convertSize(flow.size), uint8_t(0));

    std::vector<cv::Point2d> plane_list;
    std::vector<cv::Point> current_fringe;

    for (int dx = -min_max_dx; dx <= min_max_dx; ++dx) {
        for (int dy = -min_max_dx; dy <= min_max_dx; ++dy) {
            cv::Point candidate = initial + cv::Point(dx, dy);
            if (isValidIndex(candidate, flow)
                    && GT::isValidFlow(flow.at<cv::Vec2f>(candidate))
                    && edges(candidate) == 0) {
                plane_img(candidate) = 255;
                visited_img(candidate) = 255;
                sub.addPoint(candidate, cv::Point2d(candidate) + cv::Point2d(flow.at<cv::Vec2f>(candidate)));
            }
            else {
                return false;
            }
            if (std::abs(dx) == min_max_dx || std::abs(dy) == min_max_dx) {
                current_fringe.push_back(candidate);
            }
        }
    }


    sub.solve();

    current_flow = sub.result;
    std::cout << sub.summary.FullReport() << std::endl;

    double max_err = maxError(current_flow, sub.src, sub.dst);
    if (max_err >= same_plane_threshold) {
        return false;
    }

    size_t counter = 0;
    cv::Rect roi(0, 0, flow.cols, flow.rows);

    std::vector<cv::Point> next_fringe;
    for (cv::Point const& cf : current_fringe) {
        for (cv::Point const& neighbour : eight_neighbours) {
            cv::Point const current = cf + neighbour;
            if (plane_img(current) == 0 && visited_img(current) == 0) {
                next_fringe.push_back(current);
                visited_img(current) = 255;
            }
        }
    }

    for (cv::Point const& nf : next_fringe) {
        visited_img(nf) = 0;
    }

    cv::Mat_<uint8_t> bad_pixels(GT::convertSize(flow.size), uint8_t(0));
    size_t iteration_counter = 0;
    ParallelTime t;
    RunningStats solve_times, check_times, fringe_times, paint_times, iteration_times;
    while (!next_fringe.empty()) {
        ParallelTime total_time;
        t.start();

        iteration_counter++;

        current_fringe = next_fringe;
        next_fringe.clear();

        current_flow = sub.result;

        for (cv::Point const& current : current_fringe) {
            if (isValidIndex(current, flow)
                    && GT::isValidFlow(flow.at<cv::Vec2f>(current))
                    && edges(current) == 0
                    && bad_pixels(current) == 0) {
                if (plane_img(current) == 0) {
                    cv::Point2d const current_dst = cv::Point2d(current) + cv::Point2d(flow.at<cv::Vec2f>(current));
                    cv::Point2d const current_prediction = current_flow.apply(current);
                    cv::Point2d const residual = current_prediction - current_dst;
                    double const error = std::sqrt(residual.dot(residual));
                    if (error < same_plane_threshold) {
                        counter++;
                        plane_img(current) = 255;

                        sub.addPoint(current, current_dst, -1);
                    }
                    else {
                        bad_pixels(current) = 255;
                    }
                }
            }
            else {
                bad_pixels(current) = 255;
            }
        }

        std::string const check_time = t.printms();
        check_times.push(t.realTime()*1000);

        t.start();

        for (cv::Point const& current : current_fringe) {
            if (plane_img(current) == 255) {
                for (cv::Point const & neighbour : eight_neighbours) {
                    cv::Point const candidate = current + neighbour;
                    if (isValidIndex(candidate, flow)
                            && GT::isValidFlow(flow.at<cv::Vec2f>(candidate))
                            && edges(candidate) == 0
                            && plane_img(candidate) == 0
                            && bad_pixels(candidate) == 0
                            && visited_img(candidate) == 0) {
                        next_fringe.push_back(candidate);
                        visited_img(candidate) = 255;
                    }
                }
            }
        }

        std::string const fringe_time = t.printms();
        fringe_times.push(t.realTime()*1000);

        t.start();
        /*
        if (sub.get_num_points() < 1000) {
            sub.solve();
        }
        */
        bool const solved = sub.conditional_solve();
        std::string const solve_time = t.printms();
        current_flow = sub.result;
        //std::cout << sub.summary.FullReport() << std::endl;
        max_err = maxError(current_flow, sub.src, sub.dst);

        solve_times.push(t.realTime()*1000);
        t.start();
        if (solved) {
            findPlaneDebugImage2(edges, bad_pixels, plane_img, current_fringe, next_fringe);
        }
        std::string const paint_time = t.printms();
        paint_times.push(t.realTime()*1000);

        std::cout << "Iteration #" << iteration_counter << std::endl;
        std::cout << "Max error: " << max_err << std::endl;
        std::cout << "Paint time: " << paint_time << std::endl;
        std::cout << "Solve time: " << solve_time << std::endl;
        std::cout << "Check time: " << check_time << std::endl;
        std::cout << "Fringe time: " << fringe_time << std::endl;
        std::cout << "#current_fringe: " << current_fringe.size() << std::endl;
        std::cout << "#next_fringe: " << next_fringe.size() << std::endl;
        std::cout << "#flow points: " << sub.get_num_points() << std::endl;

        std::string const iteration_time = total_time.print();
        iteration_times.push(total_time.realTime()*1000);
        std::cout << "Total time: " << iteration_time << std::endl;

        std::cout << std::endl;
    }
    sub.solve();
    findPlaneDebugImage2(edges, bad_pixels, plane_img, current_fringe, next_fringe);

    std::cout << "Iterations: " << iteration_counter << ", valid: " << counter << std::endl;

    max_err = maxError(current_flow, sub.src, sub.dst);
    std::cout << "Max error: " << max_err << std::endl;

    std::cout << "iteration times: " << iteration_times.print() << std::endl;
    std::cout << "paint times: " << paint_times.print() << std::endl;
    std::cout << "solve times: " << solve_times.print() << std::endl;
    std::cout << "check times: " << check_times.print() << std::endl;
    std::cout << "fringe times: " << fringe_times.print() << std::endl;
    std::cout << "total time: " << function_time.print() << std::endl;
    return true;
}

template<class Flow>
double FlowMetrics::maxError(
        const Flow &flow,
        const std::vector<cv::Point2d> &src,
        const std::vector<cv::Point2d> &dst) const {
    double max_err = 0;
    if (src.size() != dst.size()) {
        throw std::runtime_error(std::string("src and dst vector sizes don't match in FlowMetrics::maxError, file: ") +
                                 std::string(__FILE__) + ", line " +
                                 std::to_string(__LINE__) + ", src size is " + std::to_string(src.size()) +
                                 ", dst size is " + std::to_string(dst.size())
                                 );
    }
    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        cv::Point2d const residual = flow.apply(src[ii]) - dst[ii];
        double const err = residual.dot(residual);
        max_err = std::max(max_err, err);
    }

    return std::sqrt(max_err);
}

void FlowMetrics::testPlane() {
    findPlane(gt, filtered_edge_mask, cv::Point2i(179, 101));
}

void FlowMetrics::testPlane2() {
    findArea<uint8_t, HomographyFlow>(gt, filtered_edge_mask, cv::Point2i(179, 101));
}


template<class Flow>
double FlowMetrics::flowError(
        const Flow &flow,
        const cv::Point2d &src,
        const cv::Point2d &dst) const {
    cv::Point2d const residual = flow.apply(src) - dst;
    return std::sqrt(residual.dot(residual));
}

std::string FatteningResult::printStats() {
    std::stringstream out;

    out << "#eval: " << invalidPixels.getTotalCount()
        << ", #edge_fat_bad: " << edgeFattening.getTrueCount() << " (" << edgeFattening.getPercent() << "%)"
        << ", #good: " << edgeFattening.getFalseCount() << " (" << 100.0 - edgeFattening.getPercent() << "%)";

    return out.str();
}


template<class SEG>
bool FlowMetrics::benchmarkFindFineStructure(
        const cv::Mat_<SEG> &edges,
        const cv::Point center,
        bool const debug) {
    ParallelTime time;

    int const window_rad = 10;
    int const window_size = 2*window_rad+1;

    cv::Mat_<cv::Vec3b> visu(window_size, window_size, cv::Vec3b(0,0,0));

    for (int dii = -window_rad; dii <= window_rad; ++dii) {
        for (int djj = -window_rad; djj <= window_rad; ++djj) {
            cv::Point const local_point(window_rad + dii, window_rad + djj);
            cv::Point const global_point = center + cv::Point(dii, djj);
            if (edges(global_point) == 0) {
                if (findFineStructure(edges, global_point, false)) {
                    visu(local_point) = cv::Vec3b(0,255,0);
                }
                else {
                    visu(local_point) = cv::Vec3b(0,0,255);
                }
            }
            else {
                visu(local_point) = cv::Vec3b(255, 255, 255);
            }
        }
    }

    if (debug) {
        double const realTime = time.realTime();
        std::cout << "Time: " << time.print() << std::endl;
        double num_eval_pix = window_size * window_size;
        double num_global_pix = edges.rows * edges.cols;
        std::cout << "estimated total time: " << realTime * num_global_pix / num_eval_pix << "s" << std::endl;
        cv::namedWindow("fine structure success", cv::WINDOW_GUI_EXPANDED);
        cv::imshow("fine structure success", visu);
    }

    return true;
}

template<class SEG>
bool FlowMetrics::findFineStructure(
        cv::Mat_<SEG> const& edges,
        cv::Point const center,
        bool const debug) {
    cv::Mat_<uint8_t> structures(edges.rows, edges.cols, uint8_t(0));
    return findFineStructureFast(edges, center, structures, debug);
}

template<class SEG>
bool FlowMetrics::findFineStructureComplex(
        cv::Mat_<SEG> const& edges,
        cv::Point const center,
        bool const debug) {
    ParallelTime time;
    size_t const min_segment_size = 0;
    if (edges(center) != 0) {
        return false;
    }
    cv::Mat_<uint8_t> segments(fine_structure_threshold*2+1, fine_structure_threshold*2+1, 255);
    cv::Point const segment_center(fine_structure_threshold, fine_structure_threshold);

    size_t edge_px_counter = 0;
    for (int dii = -fine_structure_threshold; dii <= fine_structure_threshold; ++dii) {
        for (int djj = -fine_structure_threshold; djj <= fine_structure_threshold; ++djj) {
            if (dii*dii + djj*djj <= fine_structure_threshold*fine_structure_threshold) {
                cv::Point const dji(djj, dii);
                cv::Point const current_point = center + dji;
                cv::Point const segment_point = segment_center + dji;
                if (isValidIndex(current_point, edges) && edges(current_point) != 0) {
                    segments(segment_point) = 254;
                    edge_px_counter++;
                }
            }
        }
    }
    // It's impossible to divide a window into three 8-connected neighbourhoods using less than 6 edge pixels.
    if (edge_px_counter < 6 && !debug) {
        return false;
    }

    std::vector<std::vector<cv::Point> > segments_points;
    segments_points.reserve(32);

    segments(fine_structure_threshold, fine_structure_threshold) = 0;
    std::stack<cv::Point> fill_stack;
    fill_stack.push(cv::Point(0,0));
    uint8_t segment_id = 0;
    bool failure = false;
    std::vector<cv::Point> segment_vec;
    while (!fill_stack.empty() && !failure) {
        cv::Point const candidate = fill_stack.top();
        fill_stack.pop();
        for (cv::Point const & neighbour : nine_neighbours) {
            cv::Point const offset = candidate + neighbour;
            cv::Point const current_point = center + offset;
            cv::Point const segment_point = segment_center + offset;
            if (
                    offset.dot(offset) <= fine_structure_threshold*fine_structure_threshold
                    && isValidIndex(current_point, edges)
                    && isValidIndex(segment_point, segments)) {
                if (edges(current_point) == 0 && segments(segment_point) == 255) {
                    segments(segment_point) = segment_id;
                    fill_stack.push(candidate + neighbour);
                    segment_vec.push_back(current_point);
                }
            }
        }
        if (fill_stack.empty()) {
            bool success = false;
            segments_points.resize(segment_id+1);
            segments_points[segment_id] = segment_vec;
            segment_id++;
            if (segment_id == 254) {
                failure = true;
            }
            segment_vec.clear();
            for (int dii = -fine_structure_threshold; !success && dii <= fine_structure_threshold; ++dii) {
                for (int djj = -fine_structure_threshold; !success && djj <= fine_structure_threshold; ++djj) {
                    cv::Point const offset(djj, dii);
                    if (offset.dot(offset) <= fine_structure_threshold*fine_structure_threshold) {
                        cv::Point const current_point = center + offset;
                        cv::Point const segment_point = segment_center + offset;
                        if (isValidIndex(current_point, edges) && segments(segment_point) == 255 && edges(current_point) == 0) {
                            success = true;
                            fill_stack.push(offset);
                        }
                    }
                }
            }
        }
    }
    if (failure && !debug) {
        return false;
    }
    if (segment_id < 2 && !debug) {
        return false;
    }
    bool found_fine_structure = false;

    std::vector<size_t> neighbour_candidates;
    for (size_t ii = 1; ii < segments_points.size(); ++ii) {
        if (segments_points[ii].size() >= min_segment_size
                &&max_neighbour_dist >= naiveNearestNeighbour(segments_points[0], segments_points[ii], max_neighbour_dist)) {
            neighbour_candidates.push_back(ii);
        }
    }

    for (size_t ii = 0; !found_fine_structure && ii < neighbour_candidates.size(); ++ii) {
        for (size_t jj = ii+1; !found_fine_structure && jj < neighbour_candidates.size(); ++jj) {
            if (min_neighbour_neighbour_dist <= naiveNearestNeighbour(
                        segments_points[neighbour_candidates[ii]],
                        segments_points[neighbour_candidates[jj]],
                        min_neighbour_neighbour_dist)) {
                found_fine_structure = true;
            }
        }
    }


    if (debug) {
        std::cout << "fine structure computation time: " << time.print() << std::endl;
        std::cout << "Estimated total time: " << time.realTime() * edges.cols * edges.rows;
        ParallelTime coloring_time;
        cv::Mat_<cv::Vec3b> colored = colorSegmentation(segments);
        cv::resize(colored, colored, cv::Size(), 15, 15, cv::INTER_NEAREST);
        std::cout << "Coloring time: " << coloring_time.print() << std::endl;
        cv::namedWindow("fine structure segmentation");
        cv::imshow("fine structure segmentation", colored);
    }

    if (failure) {
        return false;
    }
    if (segment_id < 2) {
        return false;
    }


/*
    if (found_fine_structure) {
        for (cv::Point const& p : segments_points[0]) {
            structures(p) = 255;
        }
    }
    */
    return found_fine_structure;
}

template<class SEG, class STRUCT>
void FlowMetrics::findFineStructureGlobal(
        cv::Mat_<SEG> const& edges,
        cv::Mat_<STRUCT> & result,
        bool debug) {
    ParallelTime time;
    if (edges.rows != result.rows || edges.cols != result.cols) {
        result = cv::Mat_<STRUCT>(edges.rows, edges.cols, STRUCT(0));
    }
    cv::Mat_<cv::Vec3b> visu;
    if (debug) {
        visu = cv::Mat_<cv::Vec3b>(edges.rows, edges.cols, cv::Vec3b(0,0,0));
    }
    RunningStats stats;
    cv::Mat_<uint8_t> failed(edges.rows, edges.cols, uint8_t(0));

    fine_detection_process_visu = cv::Mat_<cv::Vec3b>(edges.rows, edges.cols, cv::Vec3b(255,255,255));
    fine_detection_process_visu = setTo(fine_detection_process_visu, edges, cv::Vec3b(0,0,0));
    fine_detection_simple_visu = fine_detection_process_visu.clone();

    for (int ii = 0; ii < edges.rows; ii++) {
        for (int jj = 0; jj < edges.cols; jj++) {
            cv::Point const local_point(jj, ii);
            if (edges(local_point) == 0) {
                if (findFineStructureSimple(edges, local_point, fine_structure_threshold)) {
                    fine_detection_process_visu(local_point) = ColorsRD::yellow();
                    fine_detection_simple_visu(local_point) = ColorsRD::green();
                    if (findFineStructureComplex(edges, local_point, false)) {
                        result(local_point) = 255;
                        paintCircleToEdge(edges, result, local_point, fine_structure_threshold);
                    }
                }
            }
        }
    }
    fine_detection_process_visu = setTo(fine_detection_process_visu, result, ColorsRD::green());

    closeStructureHoles(edges, result);



    if (debug) {
        std::cout << "findFineStructureGlobal stats: " << stats.print() << std::endl;
        std::cout << "time: " << time.print() << std::endl;
        if (debug) {
            for (int ii = 0; ii < edges.rows; ++ii) {
                for (int jj = 0; jj < edges.cols; ++jj) {
                    cv::Point const point(jj, ii);
                    if (result(point) != 0) {
                        visu(point) = cv::Vec3b(0,255,0);
                    }
                    if (edges(point) != 0) {
                        visu(point) = cv::Vec3b(255,255,255);
                    }
                }
            }
        }
        cv::namedWindow("fine structure debug");
        cv::imshow("fine structure debug", visu);
    }
}

template<class SEG>
bool FlowMetrics::findFineStructureCombined(
        const cv::Mat_<SEG> &edges,
        const cv::Point center,
        const bool debug) {
    if (!findFineStructureSimple(edges, center, fine_structure_threshold-1)) {
        return false;
    }
    if (!findFineStructureComplex(edges, center, debug)) {
        return false;
    }
    return true;
}

template<class VAL>
void FlowMetrics::setBorder(cv::Mat_<VAL> &mat, int border, VAL const value) {
    for (int ii = 0; ii < border && ii < mat.rows; ++ii) {
        for (int jj = 0; jj < mat.cols; ++jj) {
            mat(ii, jj) = value;
        }
    }
    for (int ii = std::max(0, mat.rows - border); ii < mat.rows; ++ii) {
        for (int jj = 0; jj < mat.cols; ++jj) {
            mat(ii, jj) = value;
        }
    }
    for (int ii = border; ii+border < mat.rows; ++ii) {
        for (int jj = 0; jj < border && jj < mat.cols; ++jj) {
            mat(ii, jj) = value;
        }
        for (int jj = std::max(0, mat.cols - border); jj < mat.cols; ++jj) {
            mat(ii, jj) = value;
        }
    }
}
