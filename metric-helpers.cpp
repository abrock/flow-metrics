#include "metric-helpers.h"
#include "flowmetrics.h"

cv::Mat_<cv::Vec2f> fillFlowHoles(cv::Mat const& _src) {
    cv::Mat_<cv::Vec2f> src = _src.clone();
    cv::Mat_<cv::Vec2f> dst = src.clone();
    int const rows = src.rows;
    int const cols = src.cols;

    int const window = 1;
    bool found_invalid = true;
    bool found_valid = false;
    while (found_invalid) {
        found_invalid = false;
        for (int ii = 0; ii < rows; ++ii) {
            cv::Vec2f* src_l = src.ptr<cv::Vec2f>(ii);
            cv::Vec2f* dst_l = dst.ptr<cv::Vec2f>(ii);
            for (int jj = 0; jj < cols; ++jj) {
                if (GT::isValidFlow(src_l[jj])) {
                    found_valid = true;
                    continue;
                }
                found_invalid = true;
                cv::Vec2f sum (0,0);
                int counter = 0;
                for (int dii = ii - window; dii <= ii + window; dii++) {
                    for (int djj = jj - window; djj <= jj + window; djj++) {
                        if (dii > 0 && dii < rows) {
                            cv::Vec2f* line = src.ptr<cv::Vec2f>(dii);
                            if (djj > 0 && djj < cols) {
                                if (GT::isValidFlow(line[djj])) {
                                    sum += line[djj];
                                    counter++;
                                }
                            }
                        }
                    }
                }
                if (counter > 0) {
                    dst_l[jj] = sum / counter;
                }
            }
        }
        if (!found_valid) {
            return cv::Mat(rows, cols, CV_32FC2, cv::Scalar(0,0));
        }
        src = dst.clone();
    }
    return dst;
}

cv::Vec2f abs(cv::Vec2f const src) {
    return cv::Vec2f(std::abs(src[0]), std::abs(src[1]));
}

cv::Vec2f max(cv::Vec2f const a, cv::Vec2f const b) {
    return cv::Vec2f(std::max(a[0], b[0]), std::max(a[1], b[1]));
}

cv::Vec2f absmax(cv::Vec2f const& a, cv::Vec2f const& b) {
    return max(abs(a), abs(b));
}

cv::Mat fill_unc(cv::Mat const& _src) {
    cv::Mat src = _src.clone();
    cv::Mat dst = src.clone();
    int const rows = src.rows;
    int const cols = src.cols;

    int const window = 1;
    bool found_invalid = true;
    while (found_invalid) {
        found_invalid = false;
        for (int ii = 0; ii < rows; ++ii) {
            cv::Vec2f* src_l = src.ptr<cv::Vec2f>(ii);
            cv::Vec2f* dst_l = dst.ptr<cv::Vec2f>(ii);
            for (int jj = 0; jj < cols; ++jj) {
                if (GT::isValidFlow(src_l[jj])) {
                    continue;
                }
                found_invalid = true;
                cv::Vec2f local_max (0,0);
                int counter = 0;
                for (int dii = ii - window; dii <= ii + window; dii++) {
                    for (int djj = jj - window; djj <= jj + window; djj++) {
                        if (dii > 0 && dii < rows) {
                            cv::Vec2f* line = src.ptr<cv::Vec2f>(dii);
                            if (djj > 0 && djj < cols) {
                                if (GT::isValidFlow(line[djj])) {
                                    local_max = absmax(local_max, line[djj]);
                                    counter++;
                                }
                            }
                        }
                    }
                }
                if (counter > 0) {
                    dst_l[jj] = local_max;
                }
            }
        }
        src = dst.clone();
    }
    return dst;
}

void ScharrVec2f(
        cv::Mat const& src,
        cv::Mat & dst_x,
        cv::Mat & dst_y,
        int const scale,
        int const delta,
        int const border) {
    int const rows = src.rows;
    int const cols = src.cols;
    if (rows != dst_x.rows || cols != dst_x.cols) {
        dst_x = cv::Mat(rows, cols, CV_32FC2);
    }
    if (rows != dst_y.rows || cols != dst_y.cols) {
        dst_y = cv::Mat(rows, cols, CV_32FC2);
    }
    cv::Mat u(rows, cols, CV_32F), grad_u_x(rows, cols, CV_32F), grad_u_y(rows, cols, CV_32F);
    cv::Mat v(rows, cols, CV_32F), grad_v_x(rows, cols, CV_32F), grad_v_y(rows, cols, CV_32F);

    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const _src = src.ptr<cv::Vec2f>(ii);
        float * _u = u.ptr<float>(ii);
        float * _v = v.ptr<float>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            _u[jj] = _src[jj][0];
            _v[jj] = _src[jj][1];
        }
    }
    int const ddepth = CV_32F;
    cv::Scharr(u, grad_u_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT);
    cv::Scharr(u, grad_u_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);

    cv::Scharr(v, grad_v_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT);
    cv::Scharr(v, grad_v_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);

    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f * _dst_x = dst_x.ptr<cv::Vec2f>(ii);
        cv::Vec2f * _dst_y = dst_y.ptr<cv::Vec2f>(ii);

        float const * const u_x = grad_u_x.ptr<float>(ii);
        float const * const u_y = grad_u_y.ptr<float>(ii);
        float const * const v_x = grad_v_x.ptr<float>(ii);
        float const * const v_y = grad_v_y.ptr<float>(ii);

        for (int jj = 0; jj < cols; ++jj) {
            _dst_x[jj] = cv::Vec2f(u_x[jj], v_x[jj]);
            _dst_y[jj] = cv::Vec2f(u_y[jj], v_y[jj]);
        }
    }
}

cv::Mat flowGradientMagnitudes(cv::Mat const& src) {
    int const rows = src.rows;
    int const cols = src.cols;

    cv::Mat grad_x(rows, cols, CV_32FC2);
    cv::Mat grad_y(rows, cols, CV_32FC2);
    int const scale = 1;
    int const delta = 0;
    ScharrVec2f(src, grad_x, grad_y, scale, delta, cv::BORDER_DEFAULT);

    cv::Mat result(rows, cols, CV_32F);

    for (int ii = 0; ii < rows; ++ii) {
        cv::Vec2f const * const g_x = grad_x.ptr<cv::Vec2f>(ii);
        cv::Vec2f const * const g_y = grad_y.ptr<cv::Vec2f>(ii);
        float * res = result.ptr<float>(ii);
        for (int jj = 0; jj < cols; ++jj) {
            res[jj] = std::sqrt(
                        g_x[jj][0] * g_x[jj][0] +
                    g_x[jj][1] * g_x[jj][1] +

                    g_y[jj][0] * g_y[jj][0] +
                    g_y[jj][1] * g_y[jj][1]);
        }
    }
    return result;
}


std::vector<fs::path> crawl_dir_rec(fs::path const& path) {
    std::vector<fs::path> result;
    for (boost::filesystem::recursive_directory_iterator end, dir(path); dir != end; ++dir) {
        if (fs::is_regular_file(*dir)) {
            result.push_back(*dir);
        }
    }
    return result;
}

std::vector<fs::path> crawl_dir_rec(std::string const path) {
    return crawl_dir_rec(fs::path(path));
}

void evaluateMetrics(
        std::string const gt_id,
        cv::Mat const& gt,
        cv::Mat const& submission,
        cv::Mat const& unc,
        cv::Mat const& valid_mask,
        cv::Mat const& fattening_mask,
        fs::path const& dst_dir,
        json::JSON& json_result,
        json::JSON& metrics_definitions,
        bool const all_metrics) {


    if (all_metrics) {
        FlowMetrics f;
        f.evaluate(gt_id, gt, submission, unc, dst_dir, json_result, metrics_definitions);
    }
    if (true) {
        GT gt_obj;
        gt_obj.evaluate(gt_id, gt, submission, unc, valid_mask, fattening_mask, dst_dir, json_result, metrics_definitions);
    }

}

bool filenames_match(
        fs::path const& gt_dir,
        fs::path const& gt_name,
        fs::path const& submission_dir,
        fs::path const& submission) {
    fs::path relative_gt         = fs::relative(gt_name, gt_dir);
    fs::path relative_submission = fs::relative(submission, submission_dir);

    relative_gt         = fs::change_extension(relative_gt, "");
    relative_submission = fs::change_extension(relative_submission, "");

    if (relative_gt.string() == relative_submission.string()) {
        return true;
    }

    if (relative_gt.filename() == relative_submission.filename()
            && relative_gt.parent_path().filename() == relative_submission.parent_path().filename()) {
        return true;
    }

    return false;
}

bool filenames_match(
        fs::path const& gt_name,
        fs::path const& submission) {
    fs::path clean_gt         = fs::change_extension(gt_name, "");
    fs::path clean_submission = fs::change_extension(submission, "");

    if (clean_gt.filename() == clean_submission.filename()
            && clean_gt.parent_path().filename() == clean_submission.parent_path().filename()) {
        return true;
    }

    return false;
}

std::map<fs::path, fs::path> matchFiles(
        fs::path const& src_dir,
        std::vector<fs::path> const& src,
        fs::path const& dst_dir,
        std::vector<fs::path> const& dst
        ) {
    std::map<fs::path, fs::path> result;

    for (fs::path const& a : src) {
        for (fs::path const& b : dst) {
            if (filenames_match(src_dir, a, dst_dir, b)) {
                result[a] = b;
                break;
            }
        }
    }

    return result;
}

template<class A, class B>
bool is_inverse_function(std::map<A, B> const& forward, std::map<B, A> const& reverse) {
    if (forward.size() != reverse.size()) {
        return false;
    }
    for (const std::pair<A, B>& f : forward) {
        typename std::map<B, A>::const_iterator it = reverse.find(f.second);
        if (reverse.end() == it) {
            return false;
        }
        if (it->second != f.first) {
            return false;
        }
    }
    return true;
}
template bool is_inverse_function(std::map<int, int> const& forward, std::map<int, int> const& reverse);
template bool is_inverse_function(std::map<fs::path, fs::path> const& forward, std::map<fs::path, fs::path> const& reverse);

std::string ltrim_dot(std::string const& in) {
    size_t pos = 0;
    while (pos < in.length() && in[pos] == '.') {
        pos++;
    }
    return in.substr(pos);
}

std::vector<fs::path> filter_by_ext(std::vector<fs::path> const& input, std::vector<std::string> const& valid_extensions) {
    std::vector<fs::path> result;
    for (fs::path const & p : input) {
        std::string extension = ltrim_dot(p.extension().string());
        for (std::string const& cmp : valid_extensions) {
            if (ltrim_dot(cmp) == extension) {
                result.push_back(p);
                break;
            }
        }
    }
    return result;
}

std::string runEvaluationSingleGtMultipleAlgos(
        std::string const& gt_file,
        std::string const& uncFile,
        std::vector<std::string> const& submissionFiles,
        bool const save_all_images) {
    std::map<std::string, json::JSON> jsonResultsMap;
    json::JSON metrics_definitions = json::Object();

    metrics_definitions["challenge"] = "rob_flow";


    std::stringstream msg;
    if (!fs::is_regular_file(gt_file)) {
        msg << "GT file " << gt_file << " is not a regular file, aborting" << std::endl;
        throw std::runtime_error(msg.str());
    }

    cv::Mat gt_mat = GT::readOpticalFlow(gt_file);
    if (gt_mat.cols == 0 || gt_mat.rows == 0) {
        throw std::runtime_error(std::string("GT file ") + gt_file + " is damaged");
    }
    cv::Mat unc_mat, valid_mask, fattening_mask;
    if (fs::is_regular_file(uncFile)) {
        unc_mat = GT::readUncertainty(uncFile);
    }
    else {
        unc_mat = cv::Mat(gt_mat.size(), CV_32FC2);
        unc_mat.setTo(cv::Scalar(.2, .2));
    }

    FlowMetrics flowMetrics;
    flowMetrics.init(gt_mat, unc_mat);
    flowMetrics.color_scale = GT::maxFlowLength(gt_mat);
    if (save_all_images) {
        flowMetrics.prepare_images();
        cv::imwrite(gt_file + "-filled.png", flowMetrics.filled_gt_visu);
        cv::imwrite(gt_file + "-orig.png", flowMetrics.colored_gt);
        cv::imwrite(gt_file + "-propagated-gt-edges.png", flowMetrics.colored_propagated_growed_gt_edges);
        cv::imwrite(gt_file + "-propagated-gt-with-edges.png", flowMetrics.colored_propagated_growed_gt_with_edges);


        cv::imwrite(gt_file + "-grad-mag.png", flowMetrics.grad_mag_visu);
        cv::imwrite(gt_file + "-adaptive-threshold.png", flowMetrics.adaptive_threshold_visu);
        cv::imwrite(gt_file + "-unfiltered-edges-grown.png", flowMetrics.unfiltered_edges_grown);
        cv::imwrite(gt_file + "-unfiltered-edges.png", flowMetrics.unfiltered_edges);
        cv::imwrite(gt_file + "-fine-structures.png", flowMetrics.fine_structures);
        cv::imwrite(gt_file + "-fine-structure-neighbours.png", flowMetrics.fine_structure_neighbours);
        cv::imwrite(gt_file + "-filtered-edges.png", flowMetrics.filtered_edge_mask);
        cv::imwrite(gt_file + "-edge-distances.png", flowMetrics.l2distance2edges_visu);
        cv::imwrite(gt_file + "-edges-grown.png", flowMetrics.filtered_edge_mask_grown);
        cv::imwrite(gt_file + "-fine-structures-and-neighbours.png", flowMetrics.fine_structure_mask_visu);
        cv::imwrite(gt_file + "-propagated-gt.png", flowMetrics.colored_propagated_growed_gt);
        cv::imwrite(gt_file + "-propagated-distinguishable-gt.png", flowMetrics.colored_propagated_growed_trimmed_distinguishable_gt);
        cv::imwrite(gt_file + "-fine-structures-all.png", flowMetrics.fine_structure_mask_visu);
        cv::imwrite(gt_file + "-fine-structures.png", flowMetrics.fine_structures);
        cv::imwrite(gt_file + "-fine-structure-neighbours.png", flowMetrics.fine_structure_neighbours);
        cv::imwrite(gt_file + "-fine-structure-process.png", flowMetrics.fine_detection_process_visu);
        cv::imwrite(gt_file + "-fine-structure-simple.png", flowMetrics.fine_detection_simple_visu);
        std::cout << "Saved gt images" << std::endl;
    }
    for (std::string const& submission_file : submissionFiles) {

        if (!fs::is_regular_file(submission_file)) {
            msg << "submission file " << submission_file << " is not a regular file, aborting" << std::endl;
            throw std::runtime_error(msg.str());
        }



        cv::Mat submission_mat = GT::readOpticalFlow(submission_file);
        if (submission_mat.cols == 0 || submission_mat.rows == 0) {
            throw std::runtime_error(std::string("submission file ") + submission_file + " is damaged");
        }
        if (submission_mat.cols != gt_mat.cols || submission_mat.rows != gt_mat.rows) {
            /*
                throw std::runtime_error("Submission and GT sizes don't match for submission file " + submissionfile.filename().string()
                                         + ": " + GT::matsize(submission_mat) + " vs. " + GT::matsize(gt_mat));
                // */
            GT::adjust_size(gt_mat, submission_mat);
        }



        jsonResultsMap[submission_file] = json::Object();
        json::JSON & json_result= jsonResultsMap[submission_file];

        cv::Mat visuSparsity;
        double const sparsity = GT::getSparsityVisu(submission_mat, visuSparsity);
        json_result["scores"]["sparsity"]["value"] = sparsity*100;
        std::string sparsityFilename = submission_file + "-sparsity.png";
        std::string sparsityThumbFilename = submission_file + "-sparsity-thumb.png";
        json_result["scores"]["sparsity"]["visualization"]["thumb"] = sparsityThumbFilename;
        json_result["scores"]["sparsity"]["visualization"]["large"] = sparsityFilename;
        cv::imwrite(sparsityFilename, visuSparsity);
        cv::resize(visuSparsity, visuSparsity, cv::Size(), .2, .2, cv::InterpolationFlags::INTER_LANCZOS4);
        cv::imwrite(sparsityThumbFilename, visuSparsity);
        metrics_definitions["metrics"]["sparsity"] = {
                "category", "General",
                "description", "The percentage of invalid pixels in the submission.",
                "display_name", "Sparsity"};
        metrics_definitions["metrics"]["sparsity"]["result_table_visualization"] = {
                "is_visible", true,
                "legend", "black = algorithm result valid, red = algorithm result invalid, result interpolated for evaluation"};

        submission_mat = fillFlowHoles(submission_mat);

        cv::Mat colored_flow = GT::colorFlow(submission_mat, flowMetrics.color_scale);
        std::string flow_map_name = submission_file + "-flow.png";
        std::string flow_map_thumb_name = submission_file + "-flow-thumb.png";
        json_result["flow_map"] = {"thumb", flow_map_thumb_name, "large", flow_map_name};
        cv::imwrite(flow_map_name, colored_flow);
        cv::resize(colored_flow, colored_flow, cv::Size(0,0), .2, .2, cv::INTER_CUBIC);
        cv::imwrite(flow_map_thumb_name, colored_flow);


        //evaluateMetrics(gt_id, gt_mat, submission_mat, unc_mat, valid_mask, fattening_mask, dst_dir, json_result, metrics_definitions,
        //                all_metrics);
        //*
        /*
        GT gt_obj;

        fs::path const submission_dir = fs::path(submission_file).parent_path();
        std::string const submission_basename = fs::path(submission_file).filename().string();

        gt_obj.evaluate(submission_basename, gt_mat, submission_mat, unc_mat, valid_mask, fattening_mask, submission_dir, json_result, metrics_definitions);
        */
        flowMetrics.evaluateSimple(submission_file, gt_mat, submission_mat, unc_mat, json_result, metrics_definitions);

        json_result["flow_map"]["with-edges"] = submission_file + "-flow-with-edges.png";
        cv::imwrite(json_result["flow_map"]["with-edges"].ToString(), flowMetrics.colored_algo);


        //cv::imwrite(gt_file + "-propagated-gt-edges.png", flowMetrics.with_edges);

        // */

        {
            std::ofstream results_json_file(submission_file + "-results.json");
            results_json_file << json_result;
            std::cout << json_result << std::endl;
        }
        {
            std::ofstream results_json_file(submission_file + "metrics_def.json");
            results_json_file << metrics_definitions;
            std::cout << metrics_definitions << std::endl;
        }
    }


    return msg.str();

}
