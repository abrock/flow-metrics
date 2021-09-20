#include "contourflow.h"

//#include "glog/logging.h"
#include <glog/logging.h>

#include <ParallelTime/paralleltime.h>
#include <runningstats/runningstats.h>
using runningstats::RunningCovariance;
using runningstats::QuantileStats;
using runningstats::RunningStats;
using runningstats::BinaryStats;

#include <fstream>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

cv::Mat_<double> ContourFlow::getFlow() {
    /*
    if (hasFlow) {
        return valFlow;
    }
    hasFlow = true;
    valFlow = getICPFlow();
    return valFlow;
    */
    return getProjectiveFlow();
}



cv::Mat_<double> ContourFlow::getCentroidFlow() {
    cv::Mat_<double> result(3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;

    // Trivial flow estimation
    cv::Point2d flowVec = b.centroid() - a.centroid();
    result(0,2) = flowVec.x;
    result(1,2) = flowVec.y;
    return result;
}

cv::Mat_<double> ContourFlow::getTopLeftFlow() {
    cv::Mat_<double> result(3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;

    // Trivial flow estimation
    cv::Point2d flowVec = b.getMin() - a.getMin();
    result(0,2) = flowVec.x;
    result(1,2) = flowVec.y;
    return result;
}

cv::Mat_<double> ContourFlow::getBottomRightFlow() {
    cv::Mat_<double> result(3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;

    // Trivial flow estimation
    cv::Point2d flowVec = b.getMax() - a.getMax();
    result(0,2) = flowVec.x;
    result(1,2) = flowVec.y;
    return result;
}

cv::Mat_<double> ContourFlow::getBottomLeftFlow() {
    cv::Mat_<double> result(3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;

    // Trivial flow estimation
    cv::Point2d flowVecA = b.getMax() - a.getMax();
    cv::Point2d flowVecB = b.getMin() - a.getMin();
    result(0,2) = flowVecA.x;
    result(1,2) = flowVecB.y;
    return result;
}

cv::Mat_<double> ContourFlow::getTopRightFlow() {
    cv::Mat_<double> result(3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;

    // Trivial flow estimation
    cv::Point2d flowVecA = b.getMax() - a.getMax();
    cv::Point2d flowVecB = b.getMin() - a.getMin();
    result(0,2) = flowVecB.x;
    result(1,2) = flowVecA.y;
    return result;
}

cv::Mat_<double> ContourFlow::getMedianAreaFlow() {
    cv::Point2d offset = b.areaMedian() - a.areaMedian();
    cv::Mat_<double> result (3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;
    result(0,2) = offset.x;
    result(1,2) = offset.y;
    return result;
}

cv::Mat_<double> ContourFlow::getCenterFlow() {
    cv::Point2d offset = b.getCenter() - a.getCenter();
    cv::Mat_<double> result (3, 3, 0.0);
    // Initialize identity projective mapping
    result(0,0) = result(1,1) = result(2,2) = 1;
    result(0,2) = offset.x;
    result(1,2) = offset.y;
    return result;
}

cv::Vec2f ContourFlow::getUncertainty() {
    if (hasUnc) {
        return valUnc;
    }
    hasUnc = true;

    //getFlow();
    ContourFeatures warpedA = a;
    warpedA.applyMatrix(valFlow);
    valUnc = warpedA.maxDist(b);
    const cv::Vec2f valUnc2 = b.maxDist(warpedA);
    valUnc[0] = 1.0 + std::max(valUnc[0], valUnc2[0]);
    valUnc[1] = 1.0 + std::max(valUnc[1], valUnc2[1]);

    return valUnc;
}

bool ContourFlow::isVisible(double const x, double const y) const {
    double const min_frame_dist = 2;
    if (x + min_frame_dist > WIDTH || y + min_frame_dist > HEIGHT) {
        return false;
    }
    if (x < min_frame_dist || y < min_frame_dist) {
        return false;
    }
    return true;
}

bool ContourFlow::isVisible(const cv::Point2d &p) const {
    return isVisible(p.x, p.y);
}

void ContourFlow::drawFlow(
        cv::Mat_<cv::Vec2f>& flowImg,
        cv::Mat_<cv::Vec2f>& uncImg) {
    getFlow();
    getUncertainty();

    cv::Mat_<uint8_t> mask(flowImg.rows, flowImg.cols, static_cast<uint8_t>(0));
    a.drawArea(mask, cv::Scalar(1));
    const cv::Point2d min = a.getMin();
    const cv::Point2d max = a.getMax();
    const int minRow = std::max(0, static_cast<int>(std::floor(min.y)));
    const int maxRow = std::min(flowImg.rows - 1, static_cast<int>(std::ceil(max.y)));
    const int minCol = std::max(0, static_cast<int>(std::floor(min.x)));
    const int maxCol = std::min(flowImg.cols - 1, static_cast<int>(std::ceil(max.x)));


#pragma omp parallel for
    for (int ii = minRow; ii < maxRow; ++ii) {
        cv::Vec2f* flowRow = flowImg.ptr<cv::Vec2f>(ii);
        cv::Vec2f* uncRow  = uncImg.ptr<cv::Vec2f>(ii);
        uint8_t* maskRow = mask.ptr<uint8_t>(ii);
        cv::Mat_<double> src(3, 1, 1.0);
        cv::Point2d dst;
        for (int jj = minCol; jj < maxCol; ++jj) {
            if (maskRow[jj] > 0) {
                src(0) = jj;
                src(1) = ii;
                src(2) = 1;
                src = valFlow * src;
                dst = cv::Point2d(src(0), src(1)) / src(2);
                flowRow[jj][0] = static_cast<float>(dst.x - jj);
                flowRow[jj][1] = static_cast<float>(dst.y - ii);
                uncRow[jj] = valUnc;
            }
        }
    }
}

void ContourFlow::plotContour(cv::Mat_<double>& img, const double stddev) {
    double *row;
    for (int ii = 0; ii < img.rows; ++ii) {
        row = img.ptr<double>(ii);
        for (int jj = 0; jj < img.cols; ++jj) {
            row[jj] = plotAt(cv::Point2d(jj,ii), a, stddev);
        }
    }
}

RunningStats lengths;

template<class T, class P>
T integrateLineGauss(P s, P e, double stddev = 1.0) {
    T result(0.0);
    T squareLength = (e-s).dot(e-s);
    //    if (squareLength <= 0.0) {
    //        return result;
    //    }
    T length = ceres::sqrt(squareLength);
    lengths.push(length);
    const double INTEGRATION_STEPS = 16;
    for (int ii = 1; ii < INTEGRATION_STEPS; ++ii) {
        const double t = static_cast<double>(ii) / INTEGRATION_STEPS;
        const P integrationPoint = t*e + (1-t)*s;
        const T normSquare = integrationPoint.dot(integrationPoint);
        result += ceres::exp(- normSquare / (2*stddev));
    }
    // Add the component of the two endpoints for trapez rule integration
    result += ceres::exp(- s.dot(s) / (2*stddev)) / T(2.0);
    result += ceres::exp(- e.dot(e) / (2*stddev)) / T(2.0);
    return length * result;
}

template<class T, class P>
T integrateLineSquare(P s, P e, double stddev = 1.0) {
    const T squareLength = (e-s).dot(e-s);
    if (squareLength <= T(0)) {
        return T(0);
    }
    const T length = ceres::sqrt(squareLength);
    return length * (e.x*e.x + e.y*e.y + s.x*s.x + s.y*s.y + e.x*s.x + e.y*s.y);
}

template<class T, class P>
T minimum_distance(P v, P w) {
    // Return minimum distance between line segment vw and point p
    const T l2 = (v-w).dot(v-w);  // i.e. |w-v|^2 -  avoid a sqrt
    if (l2 <= 0.0) return v.dot(v);   // v == w case
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line.
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    // We clamp t from [0,1] to handle points outside the segment vw.
    const T t = std::max(T(0), std::min(T(1), (v).dot(v-w) / l2));
    const P projection = v + t * (w - v);  // Projection falls on the segment
    return projection.dot(projection);
}

template<class T, class P>
T integrateLineLinear(P s, P e, const double stddev = 1.0) {
    T result = T(stddev) - ceres::sqrt(minimum_distance<T>(s,e,P(0,0)));
    if (result < T(0)) {
        return T(0);
    }
    return result;
    //return (std::max(T(0), T(stddev) - ceres::sqrt(minimum_distance<T>(s,e,P(0,0)))));
}

template<class T, class P>
T integrateLineDistGauss(P s, P e, const double stddev = 1.0) {
    return ceres::sqrt((e-s).dot(e-s)) * ceres::exp(minimum_distance<T>(s,e) / (2*stddev));
}

template<class T, class Contour, class P>
T integrateContour(const Contour& contourVec, const P shift, const double stddev) {
    T result(0);
    for (const auto& cont : contourVec.getLines()) {
        for (size_t ii = 1; ii < cont.size(); ++ii) {
            result += integrateLineGauss<T>(P(cont[ii-1]) - shift, P(cont[ii]) - shift, stddev);
        }
    }
    return result;
}

double ContourFlow::plotAt(const cv::Point2d p, const ContourFeatures& contour, const double stddev) {
    return integrateContour<double>(contour, p, stddev);
}

struct ProjectionCostShift {
    const ContourFeatures& target;
    const CeresPoint<double> point;
    ProjectionCostShift(
            const CeresPoint<double> _point,
            const ContourFeatures& _target): target(_target), point(_point) {}
    template<class T>
    bool operator()(const T* const shiftX, const T* const shiftY, T* residual) const {
        const CeresPoint<T> projection = target.pointContourProjection<T>
                (CeresPoint<T>(point.x + shiftX[0], point.y + shiftY[0]));
        residual[0] = projection.x - point.x;
        residual[1] = projection.y - point.y;
        return true;
    }
};

struct FlowCostFunctorShift {
    FlowCostFunctorShift(
            const CeresPoint<double> point,
            const ContourFeatures& _a,
            const ContourFeatures& _b) : p(point), a(_a) {
        residualOffset = -integrateContour<double>(_b, point, stddev);
    }
    const CeresPoint<double> p;
    const ContourFeatures& a;
    const double stddev = 100;
    double residualOffset;
    template <typename T>
    bool operator()(const T* const shiftX, const T* const shiftY, T* residual) const {
        residual[0] = T(residualOffset);
        const CeresPoint<T> shift = CeresPoint<T>(p) - CeresPoint<T>(shiftX[0], shiftY[0]);
        residual[0] += integrateContour<T>(a, shift, stddev);
        return true;
    }
};

struct ICPCost {
    const CeresPoint<double> src;
    CeresPoint<double> dst;
    CeresPoint<double> aligned;
    double alignedWeight = 0.1;
    template<class P>
    ICPCost(const P point, const ContourFeatures& c) : src(point) {
        std::tuple<P, P, P, double> proj = c.pointContourProjectionSub<double>(point);
        dst = std::get<0>(proj);
        // We need the vector aligned to the contour
        aligned = std::get<2>(proj) - std::get<1>(proj);
        const double alignedSqrLength = aligned.dot(aligned);
        if (alignedSqrLength <= 0) {
            aligned.x = 1;
            aligned.y = 0;
        }
        else {
            aligned = 1.0/std::sqrt(alignedSqrLength) * aligned;
        }
        const double t = std::get<3>(proj);
        if (t <= 0 || t >= 1) {
            alignedWeight = 1;
        }
    }

    template<class T>
    bool operator()(const T* const shiftX, const T* const shiftY, T* residual) const {
        CeresPoint<T> shift(shiftX[0], shiftY[0]);
        CeresPoint<T> res = CeresPoint<T>(dst) - (CeresPoint<T>(src) + shift);
        residual[0] = res.dot(aligned) * alignedWeight;
        residual[1] = res.dotrot(aligned);
        //residual[0] = res.x;
        //residual[1] = res.y;
        return true;
    }
};

struct SimpleICPCost {
    CeresPoint<double> const src;
    CeresPoint<double> const dst;

    template<class P>
    SimpleICPCost(const P _src, const P _dst) : src(_src), dst(_dst) {
    }

    template<class T>
    bool operator()(const T* const shiftX, const T* const shiftY, T* residual) const {
        residual[0] = dst.x - (src.x + shiftX[0]);
        residual[1] = dst.y - (src.y + shiftY[0]);
        return true;
    }
};

struct ContourFlow::ProjectiveICPCost {
    static const size_t num_residuals = 2;
    CeresPoint<double> const src;
    CeresPoint<double> const dst;

    template<class P>
    ProjectiveICPCost(const P _src, const P _dst) : src(_src), dst(_dst) {
    }

    template<class T>
    bool operator()(
            const T* const m00,
            const T* const m01,
            const T* const m02,

            const T* const m10,
            const T* const m11,
            const T* const m12,

            const T* const m20,
            const T* const m21,
            const T* const m22,

            T* residual) const {

        T const w = src.x * m20[0] + src.y * m21[0] + m22[0];
        if (ceres::abs(w) < 1e-10) {
            return false;
        }
        T const x = (src.x * m00[0] + src.y * m01[0] + m02[0]) / w;
        T const y = (src.x * m10[0] + src.y * m11[0] + m12[0]) / w;
        residual[0] = dst.x - x;
        residual[1] = dst.y - y;
        return true;
    }
};

struct ContourFlow::ProjectiveICPCostRelaxed {
    static const size_t num_residuals = 2;
    CeresPoint<double> const src;
    CeresPoint<double> const dst;
    CeresPoint<double> unit_normal;

    template<class P>
    ProjectiveICPCostRelaxed(const P _src, const P _dst) :
        src(_src),
        dst(_dst),
        unit_normal((_src - _dst) / (CeresPoint<double>((_src - _dst)).norml2())) {
    }

    template<class T>
    bool operator()(
            const T* const m00,
            const T* const m01,
            const T* const m02,

            const T* const m10,
            const T* const m11,
            const T* const m12,

            const T* const m20,
            const T* const m21,
            const T* const m22,

            T* residual) const {

        T const w = src.x * m20[0] + src.y * m21[0] + m22[0];
        if (ceres::abs(w) < 1e-10) {
            return false;
        }
        T const dx = dst.x - (src.x * m00[0] + src.y * m01[0] + m02[0]) / w;
        T const dy = dst.y - (src.x * m10[0] + src.y * m11[0] + m12[0]) / w;
        residual[0] = dx * unit_normal.x + dy * unit_normal.y;
        residual[1] = (dx * unit_normal.y - dy * unit_normal.x) * .1;
        return true;
    }
};

struct ContourFlow::RestrictedFlowCost {
    static const size_t num_residuals = 2;
    CeresPoint<double> const src;
    CeresPoint<double> const dst;

    template<class P>
    RestrictedFlowCost(const P _src, const P _dst) : src(_src), dst(_dst) {
    }

    template<class T>
    bool operator()(
            const T* const data,
            T* residual) const {

        RestrictedFlow flow;
        CeresPoint<T> warped = flow.apply(data, CeresPoint<T>(src));

        residual[0] = dst.x - warped.x;
        residual[1] = dst.y - warped.y;
        return true;
    }
};

struct ContourFlow::DispFlowCost {
    static const size_t num_residuals = 2;
    CeresPoint<double> const src;
    CeresPoint<double> const dst;

    template<class P>
    DispFlowCost(const P _src, const P _dst) : src(_src), dst(_dst) {
    }

    template<class T>
    bool operator()(
            const T* const data,
            T* residual) const {

        DispFlow flow;
        CeresPoint<T> warped = flow.apply(data, CeresPoint<T>(src));

        residual[0] = dst.x - warped.x;
        residual[1] = dst.y - warped.y;
        return true;
    }
};

struct ContourFlow::SimilarityFlowCost {
    static const size_t num_residuals = 2;
    CeresPoint<double> const src;
    CeresPoint<double> const dst;

    template<class P>
    SimilarityFlowCost(const P _src, const P _dst) : src(_src), dst(_dst) {
    }

    template<class T>
    bool operator()(
            const T* const data,
            T* residual) const {

        SimilarityFlow flow;
        CeresPoint<T> warped = flow.apply(data, src);

        residual[0] = dst.x - warped.x;
        residual[1] = dst.y - warped.y;
        return true;
    }
};


double getLibICPRotationFlowSub(
        cv::Mat_<double>& result,
        std::vector<double> const& a_points,
        std::vector<double> const& b_points,
        double const inlier_dist) {
    throw(0);
}

cv::Mat_<double> ContourFlow::getLibICPRotationFlow(size_t const num_points, double const inlier_dist) {
    std::vector<double> a_points = a.getPointArray(num_points);
    std::vector<double> b_points = b.getPointArray(num_points);

    std::vector<cv::Mat_<double> >solutions = getInitialGuesses();

    size_t best_solution = 0;

    double best_cost = std::numeric_limits<double>::max();
    for (size_t ii = 0; ii < solutions.size(); ++ii) {
        double const current_cost = getLibICPRotationFlowSub(solutions[ii], a_points, b_points, inlier_dist);
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_solution = ii;
        }
    }
    if (debugOutput) {
        std::cout << "best libicp translational: " << best_solution << ", cost: " << best_cost << std::endl;
    }

    a_points = a.getPointArray(10 * num_points);
    b_points = b.getPointArray(10 * num_points);
    getLibICPRotationFlowSub(solutions[best_solution], a_points, b_points, inlier_dist);

    return solutions[best_solution];
}

double getLibICPTranslationFlowSub(
        cv::Mat_<double>& result,
        std::vector<double> const& a_points,
        std::vector<double> const& b_points,
        double const inlier_dist) {
    throw(0);
}

cv::Mat_<double> ContourFlow::getLibICPTranslationFlow(size_t const num_points, double const inlier_dist) {
    std::vector<double> a_points = a.getPointArray(num_points);
    std::vector<double> b_points = b.getPointArray(num_points);

    std::vector<cv::Mat_<double> >solutions = getInitialGuesses();

    cv::Mat_<double> best_solution;

    double best_cost = std::numeric_limits<double>::max();
    for (size_t ii = 0; ii < solutions.size(); ++ii) {
        double current_cost = b.meanDistance(a, solutions[ii], 1);
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_solution = solutions[ii];
            std::cout << "winner " << ii << " is unmodified" << std::endl;
        }
        getLibICPTranslationFlowSub(solutions[ii], a_points, b_points, inlier_dist);
        current_cost = b.meanDistance(a, solutions[ii], 1);
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_solution = solutions[ii];
            std::cout << "winner " << ii << " winner is modified" << std::endl;
        }
    }
    if (debugOutput) {
        std::cout << "best libicp translational: " << best_solution << ", cost: " << best_cost << std::endl;
    }

    return best_solution;
}

std::vector<cv::Mat_<double> > ContourFlow::getInitialGuesses() {
    std::vector<cv::Mat_<double> > solutions;
    solutions.push_back(getCentroidFlow());    // 0
    solutions.push_back(getCenterFlow());      // 1
    solutions.push_back(getMedianAreaFlow());  // 2
    solutions.push_back(getTopLeftFlow());     // 3
    solutions.push_back(getTopRightFlow());    // 4
    solutions.push_back(getBottomLeftFlow());  // 5
    solutions.push_back(getBottomRightFlow()); // 6
    return solutions;
}

cv::Mat_<double> ContourFlow::getBestIG() {
    if (hasBestIG) {
        return bestIG;
    }
    std::vector<cv::Mat_<double> > solutions = getInitialGuesses();
    /*
    std::vector<double> a_points = a.getPointArray(a.countNodes()*15);
    std::vector<double> b_points = b.getPointArray(b.countNodes()*15);
    double const inlier_dist = 10;
    */

    double best_cost = std::numeric_limits<double>::max();
    bool best_is_modified = false;
    size_t best_index = 0;
    for (size_t ii = 0; ii < solutions.size(); ++ii) {
        double current_cost = b.meanDistance(a, solutions[ii], 1);
        if (current_cost < best_cost) {
            best_cost = current_cost;
            bestIG = solutions[ii].clone();
            best_index = ii;
            best_is_modified = false;
        }
        /*
        getLibICPTranslationFlowSub(solutions[ii], a_points, b_points, inlier_dist);
        current_cost = b.meanDistance(a, solutions[ii], 1);
        if (current_cost < best_cost) {
            best_cost = current_cost;
            bestIG = solutions[ii].clone();
            best_index = ii;
            best_is_modified = true;
        }
        */
    }
    std::cout << "winner " << best_index << " is " << (best_is_modified ? "modified" : "unmodified") << std::endl;

    hasBestIG = true;
    return bestIG;
}

void ContourFlow::normalize(cv::Mat_<double> & mat) {
    mat /= norm(mat, CV_L2);
}

cv::Mat_<double> ContourFlow::getRestrictedFlow(const double cauchyParam, const size_t num_points, const size_t max_it)
{
    if (!hasFlow) {
        valFlow = getICPFlow();
        hasFlow = true;
    }
    if (hasRestrictedFlow) {
        return valRestrictedFlow;
    }
    RestrictedFlow result(valFlow);

    std::vector<cv::Point2d> const a_points = a.getPoints(num_points);
    std::vector<cv::Point2d> const b_points = b.getPoints(num_points);

    size_t const total_num_points = a_points.size() + b_points.size();

    size_t ii = 0;
    bool convergence = false;
    for (ii = 0; ii < max_it; ++ii) {
        //std::cout << std::endl << "Iteration #" << ii << std::endl;

        cv::Mat_<double> inverted_ig = result.getMatrix().inv();
        std::vector<cv::Point2d> src_points, dst_points;
        src_points.reserve(total_num_points);
        dst_points.reserve(total_num_points);
        for (const auto& p : a_points) {
            auto dst = b.pointContourProjectionSub<double>(result.apply(p));
            auto dst_point = std::get<0>(dst);
            if (isVisible(p) && isVisible(dst_point)) {
                src_points.push_back(p);
                dst_points.push_back(dst_point);
            }
        }
        //*
        for (const auto& p : b_points) {
            auto dst = a.pointContourProjectionSub<double>(ContourFeatures::applyMatrix(inverted_ig, p));
            auto dst_point = std::get<0>(dst);
            if (isVisible(p) && isVisible(dst_point)) {
                src_points.push_back(dst_point);
                dst_points.push_back(p);
            }
        }
        // */
        RestrictedFlow result_new = getRestrictedFlowSub(result, src_points, dst_points, cauchyParam);

        cv::Mat_<double> residual = result.getMatrix() - result_new.getMatrix();
        //std::cout << "Residual: " << std::endl << residual << std::endl;
        //std::cout << "Old: " << std::endl << result << std::endl << "new: " << std::endl << result_new << std::endl;
        double const update = cv::norm(residual, cv::NORM_INF);
        //std::cout << "update #it " << ii << ": " << update << std::endl;
        result = result_new;
        if (update < 1e-10) {
            //std::cout << "breaking at it# " << ii << std::endl;
            convergence = true;
            break;
        }
    }
    std::cout << "Converged: " << convergence << std::endl;

    hasRestrictedFlow = true;
    valRestrictedFlow = result.getMatrix();
    normalize(valRestrictedFlow);

    return valRestrictedFlow;

}

RestrictedFlow ContourFlow::getRestrictedFlowSub(const RestrictedFlow &ig, const std::vector<cv::Point2d> &src, const std::vector<cv::Point2d> &dst, const double cauchyParam)
{
    typedef RestrictedFlowCost Cost;
    RestrictedFlow result = ig;
    //FLAGS_logtostderr = 0;
    //FLAGS_v = 0;
    FLAGS_stderrthreshold = 3;

    if (src.size() != dst.size()) {
        throw std::runtime_error("src and dst size do not match in cv::Mat_<double> ContourFlow::getProjectiveFlowSub");
    }

    Problem problem;
    for (size_t ii = 0; ii < src.size(); ++ii) {
        CostFunction* flow_cost_function =
                new AutoDiffCostFunction<Cost, Cost::num_residuals, 6>
                (new Cost(src[ii], dst[ii]));
        ceres::CauchyLoss* loss = NULL;
        if (cauchyParam > 0) {
            loss = new ceres::CauchyLoss(cauchyParam);
        }
        problem.AddResidualBlock(flow_cost_function, loss,
                                 result.data
                                 );
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;

    Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl;


    return result;
}

DispFlow ContourFlow::getDispFlowSub(const DispFlow &ig, const std::vector<cv::Point2d> &src, const std::vector<cv::Point2d> &dst, const double cauchyParam)
{
    typedef DispFlowCost Cost;
    DispFlow result = ig;
    //FLAGS_logtostderr = 0;
    //FLAGS_v = 0;
    FLAGS_stderrthreshold = 3;

    if (src.size() != dst.size()) {
        throw std::runtime_error("src and dst size do not match in cv::Mat_<double> ContourFlow::getProjectiveFlowSub");
    }

    Problem problem;
    for (size_t ii = 0; ii < src.size(); ++ii) {
        CostFunction* flow_cost_function =
                new AutoDiffCostFunction<Cost, Cost::num_residuals, 3>
                (new Cost(src[ii], dst[ii]));
        ceres::CauchyLoss* loss = NULL;
        if (cauchyParam > 0) {
            loss = new ceres::CauchyLoss(cauchyParam);
        }
        problem.AddResidualBlock(flow_cost_function, loss,
                                 result.data
                                 );
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;

    Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl;


    return result;
}

cv::Mat_<double> ContourFlow::getSimilarityFlow(const double cauchyParam, const size_t num_points, const size_t max_it)
{
    if (!hasFlow) {
        valFlow = getICPFlow();
        hasFlow = true;
    }
    if (hasRestrictedFlow) {
        return valRestrictedFlow;
    }
    SimilarityFlow result(valFlow);

    std::vector<cv::Point2d> const a_points = a.getPoints(num_points);
    std::vector<cv::Point2d> const b_points = b.getPoints(num_points);

    size_t const total_num_points = a_points.size() + b_points.size();

    size_t ii = 0;
    bool convergence = false;
    for (ii = 0; ii < max_it; ++ii) {
        //std::cout << std::endl << "Iteration #" << ii << std::endl;

        cv::Mat_<double> inverted_ig = result.getMatrix().inv();
        std::vector<cv::Point2d> src_points, dst_points;
        src_points.reserve(total_num_points);
        dst_points.reserve(total_num_points);
        for (const auto& p : a_points) {
            auto dst = b.pointContourProjectionSub<double>(result.apply(p));
            auto dst_point = std::get<0>(dst);
            if (isVisible(p) && isVisible(dst_point)) {
                src_points.push_back(p);
                dst_points.push_back(dst_point);
            }
        }
        //*
        for (const auto& p : b_points) {
            auto dst = a.pointContourProjectionSub<double>(ContourFeatures::applyMatrix(inverted_ig, p));
            auto dst_point = std::get<0>(dst);
            if (isVisible(p) && isVisible(dst_point)) {
                src_points.push_back(dst_point);
                dst_points.push_back(p);
            }
        }
        // */
        SimilarityFlow result_new = getSimilarityFlowSub(result, src_points, dst_points, cauchyParam);

        cv::Mat_<double> residual = result.getMatrix() - result_new.getMatrix();
        //std::cout << "Residual: " << std::endl << residual << std::endl;
        //std::cout << "Old: " << std::endl << result << std::endl << "new: " << std::endl << result_new << std::endl;
        double const update = cv::norm(residual, cv::NORM_INF);
        //std::cout << "update #it " << ii << ": " << update << std::endl;
        result = result_new;
        if (update < 1e-10) {
            //std::cout << "breaking at it# " << ii << std::endl;
            convergence = true;
            break;
        }
    }
    std::cout << "Converged: " << convergence << std::endl;

    hasSimilarityFlow = true;
    valSimilarityFlow = result.getMatrix();
    normalize(valSimilarityFlow);

    return valSimilarityFlow;

}

SimilarityFlow ContourFlow::getSimilarityFlowSub(const SimilarityFlow &ig, const std::vector<cv::Point2d> &src, const std::vector<cv::Point2d> &dst, const double cauchyParam)
{
    typedef SimilarityFlowCost Cost;
    SimilarityFlow result = ig;
    //FLAGS_logtostderr = 0;
    //FLAGS_v = 0;
    FLAGS_stderrthreshold = 3;

    if (src.size() != dst.size()) {
        throw std::runtime_error("src and dst size do not match in cv::Mat_<double> ContourFlow::getProjectiveFlowSub");
    }

    Problem problem;
    for (size_t ii = 0; ii < src.size(); ++ii) {
        CostFunction* flow_cost_function =
                new AutoDiffCostFunction<Cost, Cost::num_residuals, 4>
                (new Cost(src[ii], dst[ii]));
        ceres::CauchyLoss* loss = NULL;
        if (cauchyParam > 0) {
            loss = new ceres::CauchyLoss(cauchyParam);
        }
        problem.AddResidualBlock(flow_cost_function, loss,
                                 result.data
                                 );
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;

    Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl;


    return result;
}


cv::Mat_<double> ContourFlow::getProjectiveFlow(const double cauchyParam,
                                                size_t const num_points,
                                                size_t const max_it) {
    if (!hasFlow) {
        valFlow = getICPFlow();
        hasFlow = true;
    }
    if (hasProjectiveFlow) {
        return valProjectiveFlow;
    }
    cv::Mat_<double> result = valFlow.clone();
    normalize(result);

    std::vector<cv::Point2d> const a_points = a.getPoints(num_points);
    std::vector<cv::Point2d> const b_points = b.getPoints(num_points);

    size_t const total_num_points = a_points.size() + b_points.size();

    size_t ii = 0;
    bool convergence = false;
    for (ii = 0; ii < max_it; ++ii) {
        //std::cout << std::endl << "Iteration #" << ii << std::endl;

        cv::Mat_<double> inverted_ig = result.inv();
        std::vector<cv::Point2d> src_points, dst_points;
        src_points.reserve(total_num_points);
        dst_points.reserve(total_num_points);
        for (const auto& p : a_points) {
            auto dst = b.pointContourProjectionSub<double>(ContourFeatures::applyMatrix(result, p));
            auto dst_point = std::get<0>(dst);
            if (isVisible(p) && isVisible(dst_point)) {
                src_points.push_back(p);
                dst_points.push_back(dst_point);
            }
        }
        //*
        for (const auto& p : b_points) {
            auto dst = a.pointContourProjectionSub<double>(ContourFeatures::applyMatrix(inverted_ig, p));
            auto dst_point = std::get<0>(dst);
            if (isVisible(p) && isVisible(dst_point)) {
                src_points.push_back(dst_point);
                dst_points.push_back(p);
            }
        }
        // */
        cv::Mat_<double> result_new = getProjectiveFlowSub(result, src_points, dst_points, cauchyParam);
        normalize(result_new);

        cv::Mat_<double> residual = result_new - result;
        //std::cout << "Residual: " << std::endl << residual << std::endl;
        //std::cout << "Old: " << std::endl << result << std::endl << "new: " << std::endl << result_new << std::endl;
        double const update = cv::norm(result_new - result, cv::NORM_INF);
        //std::cout << "update #it " << ii << ": " << update << std::endl;
        result = result_new;
        if (update < 1e-10) {
            //std::cout << "breaking at it# " << ii << std::endl;
            convergence = true;
            break;
        }
    }
    std::cout << "Converged: " << convergence << std::endl;

    normalize(result);
    hasProjectiveFlow = true;
    valProjectiveFlow = result.clone();

    return result;
}

template<class Flow>
struct ProjectionCost {
    CeresPoint<double> src, dst;
    ProjectionCost(cv::Point2d const _src, cv::Point2d const _dst): src(_src),  dst(_dst) {}

    template<class T>
    bool operator()(const T* const data, T* residual) const {
        const CeresPoint<T> projection = Flow::apply(data, src);
        if (!ceres::IsFinite(projection.x) || !ceres::IsFinite(projection.y)) {
            return false;
        }
        //std::cout << "projection: " << projection << std::endl;
        residual[0] = projection.x - dst.x;
        residual[1] = projection.y - dst.y;
        //std::cout << "residuals: " << residual[0] << ", " << residual[1] << std::endl;
        return true;
    }
};

template<class Flow>
ContourFlow::FlowFitter<Flow>::FlowFitter(
        const Flow &ig,
        const std::vector<cv::Point2d> &src,
        const std::vector<cv::Point2d> &dst,
        const double cauchyParam)
{
    result = Flow(ig);


    if (src.size() != dst.size()) {
        throw std::runtime_error("src and dst size do not match in ContourFlow::getHomographyFlowSub");
    }

    for (size_t ii = 0; ii < src.size(); ++ii) {
        addPoint(src[ii], dst[ii], cauchyParam);
    }

    options.linear_solver_type = ceres::DENSE_QR;
    //options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    /*
    double const tolerance = 1e-12;
    options.function_tolerance = tolerance;
    options.gradient_tolerance = tolerance;
    options.parameter_tolerance = tolerance;
    */

    solve();
    //std::cout << summary.BriefReport() << std::endl;
}

template<class Flow>
ContourFlow::FlowFitter<Flow>::FlowFitter(const Flow &ig) : result(ig) {
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
}

template<class Flow>
ContourFlow::FlowFitter<Flow>::FlowFitter() {}

template class ContourFlow::FlowFitter<HomographyFlow>;
template class ContourFlow::FlowFitter<Polynome0Flow>;
template class ContourFlow::FlowFitter<Polynome1Flow>;
template class ContourFlow::FlowFitter<Polynome2Flow>;
template class ContourFlow::FlowFitter<Polynome3Flow>;

template<class Flow>
void ContourFlow::FlowFitter<Flow>::addPoint(cv::Point2d const &_src,
                                                    cv::Point2d const &_dst,
                                                    double const cauchyParam) {
    typedef ProjectionCost<Flow> Cost;
    CostFunction* flow_cost_function =
            new AutoDiffCostFunction<Cost, 2, Flow::num_params>
            (new Cost(_src, _dst));
    ceres::CauchyLoss* loss = nullptr;
    if (cauchyParam > 0) {
        loss = new ceres::CauchyLoss(cauchyParam);
    }
    problem.AddResidualBlock(flow_cost_function, loss, result.data);
    num_points++;
    src.push_back(_src);
    dst.push_back(_dst);
}

template void ContourFlow::FlowFitter<HomographyFlow>::addPoint(cv::Point2d const &src,
                                                                       cv::Point2d const &dst,
                                                                       double const cauchyParam);

template<class Flow>
void ContourFlow::FlowFitter<Flow>::solve() {
    Solve(options, &problem, &summary);
    num_points_last_solve = get_num_points();
}
template void ContourFlow::FlowFitter<HomographyFlow>::solve();

template<class Flow>
bool ContourFlow::FlowFitter<Flow>::conditional_solve() {
    if (get_num_points() > (5 * num_points_last_solve)/4) {
        solve();
        return true;
    }
    return false;
}
template bool ContourFlow::FlowFitter<HomographyFlow>::conditional_solve();

template<class Flow>
size_t ContourFlow::FlowFitter<Flow>::get_num_points() const {
    return num_points;
}
template size_t ContourFlow::FlowFitter<HomographyFlow>::get_num_points() const;




template<class Flow>
Flow ContourFlow::getHomographyFlowSub(const Flow &ig,
                                       const std::vector<cv::Point2d> &src,
                                       const std::vector<cv::Point2d> &dst,
                                       const double cauchyParam) {
    FlowFitter<Flow> sub(ig, src, dst, cauchyParam);
    return sub.result;
}

template HomographyFlow ContourFlow::getHomographyFlowSub(
const HomographyFlow &ig,
const std::vector<cv::Point2d> &src,
const std::vector<cv::Point2d> &dst,
const double cauchyParam);

template ParamHomographyFlow ContourFlow::getHomographyFlowSub(
const ParamHomographyFlow &ig,
const std::vector<cv::Point2d> &src,
const std::vector<cv::Point2d> &dst,
const double cauchyParam);


template<class Cost>
cv::Mat_<double> ContourFlow::getProjectiveFlowSub(
        const cv::Mat_<double> &ig,
        const std::vector<cv::Point2d> &src,
        const std::vector<cv::Point2d> &dst,
        double const cauchyParam) {
    cv::Mat_<double> result = ig.clone();

    //FLAGS_logtostderr = 0;
    //FLAGS_v = 0;
    FLAGS_stderrthreshold = 3;

    if (src.size() != dst.size()) {
        throw std::runtime_error("src and dst size do not match in cv::Mat_<double> ContourFlow::getProjectiveFlowSub");
    }

    Problem problem;
    for (size_t ii = 0; ii < src.size(); ++ii) {
        CostFunction* flow_cost_function =
                new AutoDiffCostFunction<Cost, Cost::num_residuals, 1, 1, 1, 1, 1, 1, 1, 1, 1>
                (new Cost(src[ii], dst[ii]));
        ceres::CauchyLoss* loss = NULL;
        if (cauchyParam > 0) {
            loss = new ceres::CauchyLoss(cauchyParam);
        }
        problem.AddResidualBlock(flow_cost_function, loss,
                                 &result(0,0), &result(0,1), &result(0,2),
                                 &result(1,0), &result(1,1), &result(1,2),
                                 &result(2,0), &result(2,1), &result(2,2)
                                 );
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;

    Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl;

    return result;
}

cv::Mat_<double> ContourFlow::getICPFlow(const double cauchyParam, const size_t num_points, size_t const max_it) {
    //google::InitGoogleLogging("argv[0]");
    //FLAGS_logtostderr = 0;
    //FLAGS_v = 0;
    FLAGS_stderrthreshold = 3;

    std::vector<cv::Mat_<double> > solutions;
    solutions.push_back(getBestIG());

    std::vector<double> residuals(solutions.size());
    std::vector<Solver::Summary> summaries(solutions.size());
    std::vector<int> iterations_counter(solutions.size(), 0);

    bool hasUsableSolution = false;

    for (size_t kk = 0; kk < solutions.size(); ++kk) {
        //double last_cost = -1;
        std::vector<double> x_values, y_values, stepsizes;

        size_t ii = 0;
        for (ii = 0; ii < max_it; ++ii) {
            Problem problem;
            //std::cout << std::endl << "Iteration #" << ii << std::endl;
            double const last_dx = solutions[kk](0,2);
            double const last_dy = solutions[kk](1,2);
            cv::Point2d initial_guess(last_dx, last_dy);
            for (const auto& p : a.getPoints(num_points)) {
                auto dst = b.pointContourProjectionSub<double>(p+initial_guess);
                auto dst_point = std::get<0>(dst);
                if (isVisible(p) && isVisible(dst_point)) {
                    CostFunction* flow_cost_function =
                            new AutoDiffCostFunction<SimpleICPCost, 2, 1, 1>
                            (new SimpleICPCost(p, dst_point));
                    if (cauchyParam > 0) {
                        problem.AddResidualBlock(flow_cost_function, new ceres::CauchyLoss(cauchyParam), &solutions[kk](0,2), &solutions[kk](1,2));
                    }
                    else {
                        problem.AddResidualBlock(flow_cost_function, NULL, &solutions[kk](0,2), &solutions[kk](1,2));
                    }
                }
            }
            //*
            for (const auto& p : b.getPoints(num_points)) {
                auto dst = a.pointContourProjectionSub<double>(p-initial_guess);
                auto dst_point = std::get<0>(dst);
                if (isVisible(p) && isVisible(dst_point)) {
                    CostFunction* flow_cost_function =
                            new AutoDiffCostFunction<SimpleICPCost, 2, 1, 1>
                            (new SimpleICPCost(dst_point, p));
                    if (cauchyParam > 0) {
                        problem.AddResidualBlock(flow_cost_function, new ceres::CauchyLoss(cauchyParam), &solutions[kk](0,2), &solutions[kk](1,2));
                    }
                    else {
                        problem.AddResidualBlock(flow_cost_function, NULL, &solutions[kk](0,2), &solutions[kk](1,2));
                    }
                }
            }
            // */

            // Run the solver!
            Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.logging_type = ceres::SILENT;
            options.minimizer_progress_to_stdout = false;

            Solve(options, &problem, &summaries[kk]);
            if (summaries[kk].IsSolutionUsable()) {
                double const new_cost = summaries[kk].final_cost;
                double const new_dx = solutions[kk](0,2);
                double const new_dy = solutions[kk](1,2);
                double const diff_dx = last_dx - new_dx;
                double const diff_dy = last_dy - new_dy;
                residuals[kk] = new_cost;
                hasUsableSolution = true;
                /*
                std::cout << "Solution #" << kk << " is usable" << std::endl;
                std::cout << "last cost: " << last_cost << ", new cost: " << new_cost << ", improvement: " << last_cost - new_cost << std::endl;
                std::cout << "last dx: " << last_dx << ", new: " << new_dx << ", diff: " << diff_dx << std::endl;
                std::cout << "last dy: " << last_dy << ", new: " << new_dy << ", diff: " << diff_dy << std::endl;
*/
                double const max_diff = std::max(std::abs(diff_dx), std::abs(diff_dy));
                if (max_diff < 1e-6) {
                    break;
                }
                x_values.push_back(new_dx);
                y_values.push_back(new_dy);
                stepsizes.push_back(max_diff);
            }
            else {
                residuals[kk] = std::numeric_limits<double>::max();
                if (debugOutput) {
                    std::cerr << "Solution #" << kk << " is unusable" << std::endl;
                }
                break;
            }
            iterations_counter[kk] += summaries[kk].num_inner_iteration_steps;
        }
        if (debugOutput) {
            for (size_t jj = 0; jj < stepsizes.size(); jj++) {
                std::cout << "#it : " << jj
                          << ", dx: " << x_values[jj] - solutions[kk](0,2)
                             << ", dy: " << y_values[jj] - solutions[kk](1,2)
                                << ", diff: " << stepsizes[jj] << std::endl;
            }

            std::cout << "Initial guess #" << kk << " converged after " << ii << " (" << iterations_counter[kk] << ") Iterations with cost " << residuals[kk] << std::endl;
            //std::cout << summary.BriefReport() << "\n";
            //std::cout << "Computed flow: (" << initialShift.x << ", " << initialShift.y << ")" << std::endl;
            //std::cout << "Time: " << time.print() << std::endl;

            //std::cout << "Length stats: " << lengths.print() << std::endl;
        }
    }


    size_t bestIndex = 0;
    for (size_t ii = 1; ii < solutions.size(); ++ii) {
        if (residuals[ii] < residuals[bestIndex]) {
            bestIndex = ii;
        }
    }

    summary = summaries[bestIndex];

    if (debugOutput) {
        if (hasUsableSolution) {
            std::cout << "Got at least one usable solution: #" << bestIndex << std::endl;
        }
        else {
            std::cerr << "No usable solution found!" << std::endl;
        }
    }

    return solutions[bestIndex];
}

void ContourFlow::plotProjections(std::ostream& out, const size_t numPoints) {
    for (const auto& p: a.getPoints(numPoints)) {
        const CeresPoint<double> proj = b.pointContourProjection(p);
        out << p.x << "\t" << p.y << "\t" << proj.x - p.x << "\t" << proj.y - p.y << std::endl;
    }
}

void ContourFlow::plotProjections(const std::string filename, const size_t numPoints) {
    std::ofstream out(filename);
    plotProjections(out, numPoints);
    out.close();
}

cv::Point2d ContourFlow::getMedianProjectionFlow(const double trim) {
    const size_t numPoints = 50;
    std::vector<cv::Point2d> src = a.getPoints(numPoints);
    QuantileStats<float> offsetX, offsetY;
    for (size_t ii = 0; ii < src.size(); ++ii) {
        const cv::Point2d offset = b.pointContourProjection(src[ii]) - src[ii];
        offsetX.push(offset.x);
        offsetY.push(offset.y);
    }
    //return cv::Point2d(offsetX.getQuantile(0.5), offsetY.getQuantile(0.5));
    return cv::Point2d(offsetX.getTrimmedMean(trim), offsetY.getTrimmedMean(trim));
}

void ContourFlow::gnuplotProjections(const std::string filename, const size_t numPoints) {
    plotProjections(filename + "-proj.data", numPoints);
    a.plotContour(filename + "-a.data");
    b.plotContour(filename + "-b.data");

    double height = (getMax() -getMin()).y;
    double width = (getMax() - getMin()).x;

    const double offset = 0.1 * std::min(height, width);

    std::ofstream gpl(filename + ".gpl");
    gpl << "set term svg;"
        << "set output '" << filename + ".svg';"
        << "set object 1 rect from screen 0, 0, 0 to screen 1, 1, 0 behind;"
        << "set object 1 rect fc rgb 'white'  fillstyle solid 1.0;"
        << "set key out horiz;"
        << "set xrange [" << getMin().x - offset << " : " << getMax().x + offset << "];"
        << "set yrange [" << getMax().y + offset << " : " << getMin().y - offset << "] reverse;"
        << "plot "
        << "'" << filename << "-a.data' w vectors nohead title 'A',"
        << "'" << filename << "-b.data' w vectors nohead title 'B',"
        << "'" << filename << "-proj.data' w vectors title 'projections'";
    gpl.close();
    std::system(std::string("gnuplot " + filename + ".gpl").c_str());
}

cv::Point2d ContourFlow::getMin() {
    return a.pmin(a.getMin(), b.getMin());
}

cv::Point2d ContourFlow::getMax() {
    return a.pmax(a.getMax(), b.getMax());
}


void ContourFlow::drawCorrespondence(std::string filename) {
    ContourFeatures warpedA = a;
    warpedA.applyMatrix(valFlow);
    valUnc = warpedA.maxDist(b);
    a.plotContour(filename + "-A.data");
    b.plotContour(filename + "-B.data");
    a.plotFlow(filename + "-Flow.data", valFlow);
    warpedA.plotContour(filename + "-warped-A.data");
    std::ofstream gpl(filename + ".gpl");
    double offset = 10;
    gpl << "set term svg;"
        << "set output '" << filename + ".svg';"
        << "set object 1 rect from screen 0, 0, 0 to screen 1, 1, 0 behind;"
        << "set object 1 rect fc rgb 'white'  fillstyle solid 1.0;"
        << "set key out horiz;"
        << "set xrange [" << getMin().x - offset << " : " << getMax().x + offset << "];"
        << "set yrange [" << getMax().y + offset << " : " << getMin().y - offset << "];"
        << "plot "
        << "'" << filename << "-Flow.data' w vectors nohead title 'flow',"
        << "'" << filename << "-A.data' w vectors nohead title 'A',"
        << "'" << filename << "-B.data' w vectors nohead title 'B',"
        << "'" << filename << "-warped-A.data' w vectors nohead title 'warped A',";
    gpl.close();
    system(std::string(std::string("gnuplot ") + filename + ".gpl").c_str());
}


template cv::Mat_<double> ContourFlow::getProjectiveFlowSub<ContourFlow::ProjectiveICPCostRelaxed>(
        const cv::Mat_<double> &ig,
        const std::vector<cv::Point2d> &src,
        const std::vector<cv::Point2d> &dst,
        double const cauchyParam);

template<class T>
void RestrictedFlow::param2mat(const T data[], T matrix[]) const {
    T const & rotation = data[0];
    T const & scale = data[1];
    T const & shift_x = data[2];
    T const & shift_y = data[3];
    T const & proj_x = data[4];
    T const & proj_y = data[5];

    T const cos = scale * ceres::cos(rotation);
    T const sin = scale * ceres::sin(rotation);
    matrix[0 + 0] = matrix[3 + 1] = cos;
    matrix[0 + 1] = sin;
    matrix[3 + 0] = -sin;
    matrix[0 + 2] = shift_x;
    matrix[3 + 2] = shift_y;
    matrix[6 + 0] = proj_x;
    matrix[6 + 1] = proj_y;
    matrix[6 + 2] = 1;
}

template<class T>
void RestrictedFlow::param2mat(T matrix[]) const {
    param2mat(data, matrix);
}

// CeresPoint<U> RestrictedFlow::apply(const T*, const CeresPoint<U>&)’
// CeresPoint<U> RestrictedFlow::apply(const T data[], const CeresPoint<U> &src) {

template<class T, class U>
CeresPoint<T> RestrictedFlow::apply(const T data[], const CeresPoint<U> &src) {
    T const & rotation = data[0];
    T const & scale = data[1];
    T const & shift_x = data[2];
    T const & shift_y = data[3];
    T const & proj_x = data[4];
    T const & proj_y = data[5];

    U const w = U(proj_x) * U(src.x) + U(proj_y) * U(src.y) + U(1.0);

    U const cos = U(scale) * U(ceres::cos(rotation));
    U const sin = U(scale) * U(ceres::sin(rotation));

    return CeresPoint<U> (
                ( cos * src.x + sin * src.y + shift_x) / w,
                (-sin * src.x + cos * src.y + shift_y) / w
                );
}
template CeresPoint<double> RestrictedFlow::apply(const double data[], const CeresPoint<double> &src);

template<class T, class U>
CeresPoint<U> DispFlow::apply(const T data[], const CeresPoint<U> &src) const {
    T const & shift_x = data[0];
    T const & proj_x = data[1];
    T const & proj_y = data[2];

    //U const w = U(proj_x) * U(src.x) + U(proj_y) * U(src.y) + U(1.0);
    U const w = U(1);
    return CeresPoint<U> (
                (src.x + shift_x) / w,
                src.y / w
                );
}

//template CeresPoint<ceres::Jet<double, 6> > DispFlow::apply(const CeresPoint<ceres::Jet<double, 6> > &src) const;
//template CeresPoint<ceres::Jet<double, 4> > DispFlow::apply(const CeresPoint<ceres::Jet<double, 4> > &src) const;

//template CeresPoint<double> DispFlow::apply(const CeresPoint<double> &src) const;


//template CeresPoint<ceres::Jet<double, 6> > RestrictedFlow::apply(const CeresPoint<ceres::Jet<double, 6> > &src) const;
//template CeresPoint<ceres::Jet<double, 4> > RestrictedFlow::apply(const CeresPoint<ceres::Jet<double, 4> > &src) const;

//template CeresPoint<double> RestrictedFlow::apply(const CeresPoint<double> &src) const;

cv::Mat_<double> RestrictedFlow::getMatrix(const double data[]) const {
    cv::Mat_<double> result = cv::Mat_<double>::eye(3, 3);

    double matrix[9];
    param2mat(data, matrix);

    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            result[row][col] = matrix[3*row + col];
        }
    }

    return result;
}

cv::Mat_<double> RestrictedFlow::getMatrix() const
{
    return getMatrix(data);
}

void RestrictedFlow::rescale(const double factor)
{
    //scale  (factor * scale());
    shift_x(factor * shift_x());
    shift_y(factor * shift_y());
    proj_x (proj_x() / factor);
    proj_y (proj_y() / factor);
}

//template CeresPoint<double> RestrictedFlow::apply(const double data[], const CeresPoint<double> &src) const;
//template CeresPoint<ceres::Jet<double, 4> > RestrictedFlow::apply(const ceres::Jet<double, 4> data[], const CeresPoint<ceres::Jet<double, 4> > &src) const;


template<class T>
void SimilarityFlow::param2mat(const T data[], T matrix[]) const {
    T const & rotation = data[0];
    T const & scale = data[1];
    T const & shift_x = data[2];
    T const & shift_y = data[3];

    T const cos = scale * ceres::cos(rotation);
    T const sin = scale * ceres::sin(rotation);
    matrix[0 + 0] = matrix[3 + 1] = cos;
    matrix[0 + 1] = sin;
    matrix[3 + 0] = -sin;
    matrix[0 + 2] = shift_x;
    matrix[3 + 2] = shift_y;
    matrix[6 + 0] = 0;
    matrix[6 + 1] = 0;
    matrix[6 + 2] = 1;
}

template<class T>
void SimilarityFlow::param2mat(T matrix[]) const {
    param2mat(data, matrix);
}

template<class T, class U>
CeresPoint<T> SimilarityFlow::apply(const T data[], const CeresPoint<U> &src) const {
    T const & rotation = data[0];
    T const & scale = data[1];
    T const & shift_x = data[2];
    T const & shift_y = data[3];

    T const cos = scale * ceres::cos(rotation);
    T const sin = scale * ceres::sin(rotation);

    return CeresPoint<T> (
                ( cos * src.x + sin * src.y + shift_x),
                (-sin * src.x + cos * src.y + shift_y)
                );
}

cv::Mat_<double> SimilarityFlow::getMatrix(const double data[]) const {
    cv::Mat_<double> result = cv::Mat_<double>::eye(3, 3);

    double matrix[9];
    param2mat(data, matrix);

    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            result[row][col] = matrix[3*row + col];
        }
    }

    return result;
}

cv::Mat_<double> SimilarityFlow::getMatrix() const
{
    return getMatrix(data);
}

std::string SimilarityFlow::printDiff(const SimilarityFlow &other) const {
    std::stringstream out;
    out << "r: " << std::abs(rotation() - other.rotation())
        << ", s: " << std::abs(scale() - other.scale())
        << ", x: " << std::abs(shift_x() - other.shift_x())
        << ", y: " << std::abs(shift_y() - other.shift_y());
    return out.str();
}

std::string SimilarityFlow::print() const
{
    std::stringstream out;
    out << *this;
    return out.str();
}


std::string RestrictedFlow::print() const
{
    std::stringstream out;
    out << *this;
    return out.str();
}

void SimilarityFlow::rescale(const double factor)
{
    //scale  (factor * scale());
    shift_x(factor * shift_x());
    shift_y(factor * shift_y());
}

template CeresPoint<double> SimilarityFlow::apply(const double data[], const CeresPoint<double> &src) const;
template CeresPoint<ceres::Jet<double, 4> > SimilarityFlow::apply(const ceres::Jet<double, 4> data[], const CeresPoint<ceres::Jet<double, 4> > &src) const;



std::ostream &operator <<(std::ostream &out, const SimilarityFlow &flow) {
    std::cout << "r: " << flow.rotation()
              << ", s: " << flow.scale()
              << ", x: " << flow.shift_x()
              << ", y: " << flow.shift_y();
    return out;
}

std::ostream &operator <<(std::ostream &out, const RestrictedFlow &flow) {
    std::cout << "r: " << flow.rotation()
              << ", s: " << flow.scale()
              << ", x: " << flow.shift_x()
              << ", y: " << flow.shift_y()
              << ", a: " << flow.proj_x()
              << ", b: " << flow.proj_y();
    return out;
}

double HomographyFlow::frobeniusNorm() const
{
    double sum = 0;
    for (size_t ii = 0; ii < 9; ++ii) {
        sum += data[ii] * data[ii];
    }
    return std::sqrt(sum);
}

#include <random>

void HomographyFlow::addNoise(const double scale)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<double> uni(-1,1);
    for (size_t ii = 0; ii < 9; ++ii) {
        data[ii] += scale * uni(engine);
    }
}

cv::Mat_<double> HomographyFlow::getMatrix(const double data[]) const {
    return (cv::Mat_<double>(3,3) << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
}

cv::Mat_<double> HomographyFlow::getMatrix() const {
    return getMatrix(data);
}

template<class T, class U>
CeresPoint<T> HomographyFlow::apply(const T data[], const CeresPoint<U> &src)
{
    //*
    T const x = data[0] * src.x + data[1] * src.y + data[2];
    T const y = data[3] * src.x + data[4] * src.y + data[5];
    T const w = data[6] * src.x + data[7] * src.y + data[8];
    // */
    /*
    T const x = data[0] * src.x + data[1] * src.y + data[2];
    T const y =    T(0) * src.x + data[4] * src.y + T(0);
    T const w =    T(0) * src.x +    T(0) * src.y + data[4];
    return CeresPoint<T>(x/w, y/w);
    // */
    return CeresPoint<T>(x/w, y/w);
}

template CeresPoint<double> HomographyFlow::apply(const double data[], const CeresPoint<double> &src);
template CeresPoint<ceres::Jet<double, 9> > HomographyFlow::apply(const ceres::Jet<double, 9> data[], const CeresPoint<ceres::Jet<double, 9> > &src);

template<class T>
CeresPoint<T> HomographyFlow::apply(const CeresPoint<T> &src) const
{
    return HomographyFlow::apply(data, src);
}
template CeresPoint<double> HomographyFlow::apply(const CeresPoint<double> &src) const;

template<class T, class U>
CeresPoint<T> ParamHomographyFlow::apply(const T data[], const CeresPoint<U> &src)
{
    /*
    * A = R(θ) R(−φ) D R(φ)
    * 0: Rotation θ
    * 1: Rotation φ
    * 2: Scale λ1
    * 3: Scale λ2
    * 4: dx
    * 5: dy
    * 6: v1
    * 7: v2
    * 8: v3
    */
    CeresPoint<T> result(src);
    // First compute the inner product of the last row of the homography matrix with the input vector.
    T const w = src.x * data[6] + src.y * data[7] + data[8];

    // Second, compute R(θ) R(−φ) D R(φ) times the inner vector
    result.rotate(data[1]);
    result.x *= data[2];
    result.y *= data[3];
    result.rotate(-data[1]);
    result.rotate(data[0]);

    // Third, add the translation vector
    result.x += data[4];
    result.y += data[5];

    // Last, divide by the inner product computed in the first step.
    result.x /= w;
    result.y /= w;

    return result;
}

template CeresPoint<double> ParamHomographyFlow::apply(const double data[], const CeresPoint<double> &src);
template CeresPoint<ceres::Jet<double, 9> > ParamHomographyFlow::apply(const ceres::Jet<double, 9> data[], const CeresPoint<ceres::Jet<double, 9> > &src);

template<class T>
CeresPoint<T> ParamHomographyFlow::apply(const CeresPoint<T> &src) const
{
    return ParamHomographyFlow::apply(data, src);
}
template CeresPoint<double> ParamHomographyFlow::apply(const CeresPoint<double> &src) const;


void ParamHomographyFlow::addNoise(const double scale) {
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<double> uni(-1,1);
    for (size_t ii = 0; ii < 9; ++ii) {
        data[ii] += scale * uni(engine);
    }
}

cv::Mat_<double> ParamHomographyFlow::getMatrix(const double data[]) const {
    /**
     * @brief data
     * A = R(θ) R(−φ) D R(φ)
     * 0: Rotation θ
     * 1: Rotation φ
     * 2: Scale λ1
     * 3: Scale λ1
     * 4: dx
     * 5: dy
     * 6: v1
     * 7: v2
     * 8: v3
     */
    cv::Mat_<double> const D = (cv::Mat_<double>(2,2) << data[2], 0, 0, data[3]);
    cv::Mat_<double> const A = getRotationMatrix(data[0]) * getRotationMatrix(-data[1]) * D * getRotationMatrix(data[1]);
    cv::Mat_<double> result(3,3);
    result(0,0) = A(0,0);
    result(0,1) = A(0,1);
    result(1,0) = A(1,0);
    result(1,1) = A(1,1);

    // dx
    result(0,2) = data[4];

    // dy
    result(1,2) = data[5];

    // third row
    result(2,0) = data[6];
    result(2,1) = data[7];
    result(2,2) = data[8];

    return result;
}

std::ostream& operator << (std::ostream &out, ParamHomographyFlow const& in) {
    /*
    * A = R(θ) R(−φ) D R(φ)
    * 0: Rotation θ
    * 1: Rotation φ
    * 2: Scale λ1
    * 3: Scale λ2
    * 4: dx
    * 5: dy
    * 6: v1
    * 7: v2
    * 8: v3
    * */
    out << "θ: " << in.data[0] << ", φ: " << in.data[1] << ", λ1: " << in.data[2] << ", λ2: " << in.data[3]
        << ", dx: " << in.data[4] << ", dy: " << in.data[5] << ", v: " << in.data[6] << ", " << in.data[7] << ", " << in.data[8];
    return out;
}

cv::Mat_<double> ParamHomographyFlow::getMatrix() const
{
    return getMatrix(data);
}

template<class T, class U>
CeresPoint<T> Polynome2Flow::apply(const T data[], const CeresPoint<U> &src) {
    return CeresPoint<T>(
                data[0] + data[1] * src.x + data[2] * src.y + data[3] * src.x * src.y +
            data[4] * src.x * src.x + data[5] * src.y * src.y,

            data[6] + data[7] * src.x + data[8] * src.y + data[9] * src.x * src.y +
            data[10] * src.x * src.x + data[11] * src.y * src.y
            );
}

Polynome2Flow::Polynome2Flow()
{
    for (size_t ii = 0; ii < num_params; ++ii) {
        data[ii] = 0;
    }
    data[1] = data[8] = 1;
}

Polynome3Flow::Polynome3Flow()
{
    for (size_t ii = 0; ii < num_params; ++ii) {
        data[ii] = 0;
    }
    data[1] = data[12] = 1;
}

template<class FLOW>
FlowBase<FLOW>::~FlowBase() {

}

template<class FLOW>
FlowBase<FLOW>::FlowBase() {
}

template<class FLOW>
double FlowBase<FLOW>::getParam(size_t ii) {
    if (ii+1 > static_cast<FLOW*>(this)->num_params) throw std::runtime_error("index out of range");
    return (static_cast<FLOW*>(this)->data)[ii];
}

template<class FLOW>
void FlowBase<FLOW>::plotFlow(cv::Mat_<cv::Vec2f> &flow, const cv::Mat_<uint8_t> &mask) const {
    int const rows = flow.rows;
    int const cols = flow.cols;
    bool use_mask = (mask.rows == rows) && (mask.cols == cols);
    for (int ii = 0; ii < rows; ++ii) {
        for (int jj = 0; jj < cols; ++jj) {
            if (!use_mask || mask(ii, jj) > 0) {
                cv::Point2d const src(ii, jj);
                cv::Point2d dst = apply(src);
                flow(ii, jj) = cv::Vec2f(dst.x - src.x, dst.y - src.y);
            }
        }
    }
}

template<class FLOW>
cv::Point2d FlowBase<FLOW>::apply(const cv::Point2d &src) const {
    FLOW const * const self = static_cast<const FLOW*>(this);
    return self->apply(self->data, CeresPoint<double>(src)).getPoint2d();
}

template<class FLOW>
cv::Point2d FlowBase<FLOW>::apply(const double data[], const cv::Point2d &src) const {
    FLOW const * const self = static_cast<const FLOW*>(this);
    return self->apply(data, CeresPoint<double>(src)).getPoint2d();
}

template<class FLOW>
void FlowBase<FLOW>::addNoise(const double scale) {
    FLOW* self = static_cast<FLOW*>(this);

    std::normal_distribution<> dist{0,std::abs(scale)};

    for (size_t ii = 0; ii < self->num_params; ++ii) {
        self->data[ii] += dist(engine);
    }
}

template<class FLOW>
template<class T>
CeresPoint<T> FlowBase<FLOW>::apply(const CeresPoint<T> &src) const {
    FLOW const * const self = static_cast<FLOW const *>(this);
    return self->apply(self->data, src);
}


Polynome1Flow::Polynome1Flow() {
    for (size_t ii = 0; ii < num_params; ++ii) {
        data[ii] = 0;
    }
    data[1] = data[5] = 1;
}

template<class T, class U>
CeresPoint<T> Polynome1Flow::apply(const T data[], const CeresPoint<U> &src) {
    return CeresPoint<T>(
                data[0] + data[1] * src.x + data[2] * src.y,

            data[3] + data[4] * src.x + data[5] * src.y
            );
}


template<class T, class U>
CeresPoint<T> Polynome3Flow::apply(const T data[], const CeresPoint<U> &src) {
    /*
     * x+u = a_0  + a_1 x  + a_2 y  + a_3 xy  +  a_4 x²  +  a_5 y²  +  a_6 x²y +  a_7 xy² +  a_8 x³ +  a_9 y³
     * y+v = a_10 + a_11 x + a_12 y + a_13 xy +  a_14 x² +  a_15 y² + a_16 x²y + a_17 xy² + a_18 x³ + a_19 y³
     */
    T const x = T(src.x);
    T const y = T(src.y);
    return CeresPoint<T>(
                data[0] + data[1] * x + data[2] * y + data[3] * x*y +
            data[4] * x*x + data[5] * y*y + data[6] * x*x*y + data[7] * x*y*y + data[8] * x*x*x + data[9] * y*y*y,

            data[10] + data[11] * x + data[12] * y + data[13] * x*y +
            data[14] * x*x + data[15] * y*y + data[16] * x*x*y + data[17] * x*y*y + data[18] * x*x*x + data[19] * y*y*y
            );
}

Polynome0Flow::Polynome0Flow() {
    data[0] = data[1] = 0;
}

template<class T, class U>
CeresPoint<T> Polynome0Flow::apply(const T data[], const CeresPoint<U> &src) {
    return CeresPoint<T>(src.x + data[0], src.y + data[1]);
}

template FlowBase<HomographyFlow>::FlowBase();
template FlowBase<DispFlow>::FlowBase();
template FlowBase<RestrictedFlow>::FlowBase();
template FlowBase<ParamHomographyFlow>::FlowBase();
template FlowBase<Polynome2Flow>::FlowBase();
template FlowBase<SimilarityFlow>::FlowBase();

template CeresPoint<double> FlowBase<SimilarityFlow>::apply<double>(CeresPoint<double> const&) const;
template CeresPoint<double> FlowBase<RestrictedFlow>::apply<double>(CeresPoint<double> const&) const;
template CeresPoint<double> FlowBase<DispFlow>::apply<double>(CeresPoint<double> const&) const;

template void FlowBase<RestrictedFlow>::plotFlow(cv::Mat_<cv::Vec2f> &, cv::Mat_<uint8_t> const& ) const;


