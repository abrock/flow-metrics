#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <cmath>
#include <fstream>

#include "kahansum.h"

#include "contourfeatures.h"

#include <algorithm>

template<class P>
double dist(const P a, const P b) {
    return std::sqrt((a-b).dot(a-b));
}

void push_if_new(std::vector<cv::Point2d>& data, cv::Point2d const candidate) {
    if (data.empty()) {
        data.push_back(candidate);
        return;
    }
    if (dist(data.back(), candidate) > 1e-6) {
        data.push_back(candidate);
    }
}

std::vector<double> ContourFeatures::getPointArray(const size_t numPoints) const {
    std::vector<double> result;
    auto const tmp = getPoints(numPoints);
    for (auto const& p : tmp) {
        result.push_back(p.x);
        result.push_back(p.y);
    }
    return result;
}

std::vector<cv::Point2d> ContourFeatures::getPoints(const size_t numPoints) const {
    std::vector<cv::Point2d> result;
    result.reserve(numPoints);
    result.push_back(lineData.front().front());
    const double length = lineLength();
    const double targetLength = length / (numPoints - 1);
    for (const auto& cont : lineData) {
        push_if_new(result, cont.front());
        for (size_t ii = 1; ii < cont.size(); ++ii) {
            const cv::Point2d s = cont[ii-1];
            const cv::Point2d e = cont[ii];
            double const currentDist = dist(s,e);
            if (currentDist > targetLength) {
                const size_t localNum = static_cast<size_t>(std::ceil(currentDist / targetLength));
                for (size_t jj = 1; jj < localNum; jj++) {
                    const double t = static_cast<double>(jj) / localNum;
                    push_if_new(result, (s + t*(e-s)));
                }
            }
            push_if_new(result, cont[ii]);
        }
    }
    return result;
}

void ContourFeatures::plotContour(std::ostream& out) const {
    for (const auto& cont: lineData) {
        for (size_t ii = 1; ii < cont.size(); ++ii) {
            const cv::Point2d e = cont[ii];
            const cv::Point2d s = cont[ii-1];
            out << s.x << "\t" << s.y << "\t" << e.x - s.x << "\t" << e.y-s.y << std::endl;
        }
    }
}


void ContourFeatures::plotContour(const std::string filename) const {
    std::ofstream out(filename);
    plotContour(out);
    out.close();
}

cv::Point2d ContourFeatures::centroid() const {
    if (hasCentroid) {
        return valCentroid;
    }
    hasCentroid = true;
    calcAreaCentroid();
    calcLineCentroid();
    return valCentroid;
}

void ContourFeatures::calcAreaCentroid() const {
    if (areaData.empty()) {
        valArea = valCentroid.x = valCentroid.y = 0;
        return;
    }
    KahanSum globalArea, globalX, globalY;
    for (auto & contour : areaData) {
        if (!contour.empty() && contour.size() > 1) {
            auto s = contour.back();
            KahanSum kahanSignedArea, centroidX, centroidY;
            for (auto e : contour) {
                // Errors in area, centroid x/y:
                // 8.97415e-08 +- 1.13613e-07, 10000 Samples, range: [2.10321e-12, 1.49123e-06]
                // 4.08143e-09 +- 2.42257e-08, 10000 Samples, range: [0, 2.38419e-07]
                // 3.73498e-09 +- 2.31349e-08, 10000 Samples, range: [0, 4.76837e-07]
                centroidX.push((e.x + s.x) * (e.y * s.x - e.x * s.y));
                centroidY.push(-(e.y + s.y) * (e.x * s.y - e.y * s.x));
                kahanSignedArea.push(e.y * s.x - e.x * s.y);

                s = e;
            }
            const double valSignedArea = kahanSignedArea.getSum()/2;
            const int sign = kahanSignedArea.getSum() < 0 ? -1 : 1;

            globalArea.push(sign*valSignedArea);
            globalX   .push(sign*centroidX.getSum());
            globalY   .push(sign*centroidY.getSum());
        }
    }
    valArea = globalArea.getSum();
    if (0 >= valArea) {
        valCentroid = getCenter();
    }
    else {
        valCentroid.x = globalX.getSum();
        valCentroid.y = globalY.getSum();
        valCentroid /= 6*valArea;
    }
}


void ContourFeatures::calcLineCentroid() const {
    if (lineData.empty()) {
        valLineLength = valLineCentroid.x = valLineCentroid.y = 0;
        return;
    }
    KahanSum lineLength, lineCentroidX, lineCentroidY;

    for (const auto& contour : lineData) {
        if (!contour.empty()) {
            auto s = contour.front();
            for (auto e : contour) {
                const double segmentLength = std::sqrt((e-s).dot(e-s));
                lineLength.push(segmentLength);
                lineCentroidX.push(segmentLength * e.x);
                lineCentroidY.push(segmentLength * e.y);
                s = e;
            }
        }
    }
    valLineLength = lineLength.getSum();
    if (0 < valLineLength) {
        valLineCentroid.x = lineCentroidX.getSum()/valLineLength;
        valLineCentroid.y = lineCentroidY.getSum()/valLineLength;
    }
    else {
        valLineCentroid = getCenter();
    }
}


cv::Mat_<double> ContourFeatures::covarianceMatrix() {
    // Make sure centroid and signed area are computed.
    centroid();
    covMat = cv::Mat_<double>(2,2, 0.0);
    if (0 >= area()) {
        momentXXX = momentYYY = momentXXY = momentXYY = 0.0;
        return covMat;
    }

    KahanSum gCov, gX, gY, gXXX, gXXY, gXYY, gYYY;

    for (auto & contour : areaData) {
        auto _s = contour.back();

        KahanSum varX, varY, cov, momXXX, momYYY, momXXY, momXYY, signedArea;

        for (auto _e : contour) {
            auto s = _s - valCentroid;
            auto e = _e - valCentroid;
            // Naive implementation, Errors for varX, varY, cov:
            // 6.19396e-14 +- 7.6602e-12, 320000 Samples, range: [0, 3.73977e-09]
            // 1.20899e-13 +- 4.4359e-11, 320000 Samples, range: [0, 2.5062e-08]
            // 2.06385e-13 +- 9.42515e-11, 320000 Samples, range: [0, 5.33067e-08]
            //covMat(0,0) += 4*dy*(s.x*s.x*s.x+(3./2.)*s.x*s.x*(e.x-s.x)+s.x*(e.x-s.x)*(e.x-s.x)+(e.x-s.x)*(e.x-s.x)*(e.x-s.x)/4);
            //covMat(1,1) -= 4*dx*(s.y*s.y*s.y+(3./2.)*s.y*s.y*(e.y-s.y)+s.y*(e.y-s.y)*(e.y-s.y)+(e.y-s.y)*(e.y-s.y)*(e.y-s.y)/4);
            //covMat(0,1) += 12*dy*(s.x*s.x*s.y + s.x*dx*s.y + dy*s.x*s.x/2 + (dx*dx*s.y + 2* s.x*dx*dy)/3 + dx*dx*dy/4);


            //covMat(0,1) += ((s.x * (e.y-s.y))*(2*s.y*e.x + s.x*(e.y+3*s.y)) + (e.y-s.y)*e.x*(e.x*s.y+e.y*(2*s.x+3*e.x)));
            // Simple factorized version of the naive implementation, Errors for varX, varY, cov:
            // 3.64545e-14 +- 6.83779e-12, 320000 Samples, range: [0, 3.79179e-09]
            // 3.04036e-14 +- 3.17275e-12, 320000 Samples, range: [0, 1.56764e-09]
            // 3.53404e-14 +- 6.00203e-12, 320000 Samples, range: [0, 3.28126e-09]
            //covMat(0,0) += (e.x + s.x) * (e.y - s.y) * (e.x * e.x + s.x * s.x);
            //covMat(1,1) -= (e.y + s.y) * (e.x - s.x) * (e.y * e.y + s.y * s.y);
            //covMat(0,1) += (e.y - s.y) * (3 * e.x *e.x * e.y + e.x *e.x * s.y + 2 * e.x * e.y * s.x + 2 * e.x * s.x * s.y + e.y * s.x *s.x + 3 * s.x * s.x * s.y);

            // Making use of the telescope sum, Errors for varX, varY, cov:
            // 3.58507e-14 +- 2.65226e-12, 320000 Samples, range: [0, 8.6365e-10]
            // 3.82768e-14 +- 3.27276e-12, 320000 Samples, range: [0, 1.03131e-09]
            // 2.79821e-14 +- 2.27365e-12, 320000 Samples, range: [0, 8.55536e-10]
            // Using Kahan Summation when calculating the signed area further improves things:
            // Errors for varX, varY, cov:
            // 3.17895e-14 +- 1.86729e-12, 320000 Samples, range: [0, 7.15344e-10]
            // 3.24688e-14 +- 2.37408e-12, 320000 Samples, range: [0, 1.03672e-09]
            // 2.46674e-14 +- 1.86948e-12, 320000 Samples, range: [0, 8.61169e-10]
            // covMat(0,0) += (e.x * e.x + e.x * s.x + s.x *s.x) * (e.y * s.x - e.x * s.y);
            // covMat(1,1) -= (e.y * e.y + e.y * s.y + s.y *s.y) * (e.x * s.y - e.y * s.x);
            // covMat(0,1) += (2 * e.x * e.y + e.x * s.y + e.y * s.x + 2 * s.x * s.y) * (e.y * s.x - e.x * s.y);

            // Implementation using Kahan summation:
            // Errors for varX, varY, cov:
            // 2.74947e-14 +- 1.13048e-12, 320000 Samples, range: [0, 2.72328e-10]
            // 2.76458e-14 +- 1.19105e-12, 320000 Samples, range: [0, 4.53642e-10]
            // 2.02167e-14 +- 7.12903e-13, 320000 Samples, range: [0, 1.97986e-10]
            varX.push((e.x * e.x + s.x * e.x + s.x * s.x) * (e.y * s.x - e.x * s.y));
            varY.push((e.y * e.y + e.y * s.y + s.y * s.y) * (e.y * s.x - e.x * s.y));
            cov.push((2 * e.x * e.y + e.x * s.y + e.y * s.x + 2 * s.x * s.y) * (e.y * s.x - e.x * s.y));
            momXXX.push((e.x + s.x) * (e.x * e.x + s.x * s.x) * (e.x * s.y - e.y * s.x));
            momYYY.push((e.y + s.y) * (e.y * e.y + s.y * s.y) * (e.y * s.x - e.x * s.y));
            momXXY.push((e.x * e.x * (3 * e.y + s.y) + 2 * e.x * s.x * (e.y + s.y) + s.x * s.x * (e.y + 3 * s.y)) * (e.x * s.y - e.y * s.x));
            momXYY.push((e.y * e.y * (3 * e.x + s.x) + 2 * e.y * s.y * (e.x + s.x) + s.y * s.y * (e.x + 3 * s.x)) * (e.y * s.x - e.x * s.y));

            // We need the signed area of every sub-contour in order to determine
            // the sign of the global update.
            signedArea.push(e.y * s.x - e.x * s.y);

            // Expanded variant for testing has a much larger error:
            // 3.98034e-14 +- 2.17755e-12, 320000 Samples, range: [0, 7.53879e-10]
            // varX.push(-e.x * s.y * e.x * e.x - e.x * s.y * s.x * s.x - s.x * s.y * e.x * e.x + e.x * e.y * s.x * s.x + e.y * s.x * e.x * e.x + e.y * s.x * s.x * s.x);

            _s = _e;
        }
        const int sign = signedArea.getSum() < 0 ? -1 : 1;
        gCov.push(sign * cov.getSum());
        gX.push(sign * varX.getSum());
        gY.push(sign * varY.getSum());
        gXXX.push(sign * momXXX.getSum());
        gXXY.push(sign * momXXY.getSum());
        gXYY.push(sign * momXYY.getSum());
        gYYY.push(sign * momYYY.getSum());

    }

    covMat(0,0) = gX.getSum() / (12*valArea);
    covMat(1,1) = gY.getSum() / (12*valArea);
    covMat(0,1) = gCov.getSum() / (24*valArea);
    covMat(1,0) = covMat(0,1);

    momentXXX = gXXX.getSum() / (20 * valArea);
    momentYYY = -gYYY.getSum() / (20 * valArea);
    momentXXY = gXXY.getSum() / (60 * valArea);
    momentXYY = -gXYY.getSum() / (60 * valArea);
    return covMat;
}

template<class Time, class Point>
std::pair<Point, Time> ContourFeatures::pointLineProjectionSub(const Point v, const Point w, const Point p) const {
    // Return minimum distance between line segment vw and point p
    const Time l2 = (v-w).dot(v-w);  // i.e. |w-v|^2 -  avoid a sqrt
    if (l2 <= 0.0) return std::pair<Point, Time>(v, Time(0));   // v == w case
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line.
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    // We clamp t from [0,1] to handle points outside the segment vw.
    const Time t = std::max(Time(0), std::min(Time(1), (p-v).dot(w-v) / l2));
    return std::pair<Point, Time>(v + t * (w - v), t);  // Projection falls on the segment
}

template<class T, class P>
P ContourFeatures::pointLineProjection(const P v, const P w, const P p) const {
    return pointLineProjectionSub<T, P>(v, w, p).first();
}

template<class T, class P>
std::tuple<P, P, P, T> ContourFeatures::pointContourProjectionSub(const P p) const {
    P bestProjection = getLines().front().front();
    P bestStart = bestProjection;
    P bestEnd = getLines().front()[1];
    T bestTime = T(0);
    T bestDistance = (p-bestProjection).dot(p-bestProjection);
    for (const auto& cont : getLines()) {
        for (size_t ii = 1; ii < cont.size(); ++ii) {
            const std::pair<P, T> currentResult = pointLineProjectionSub<T, P>(cont[ii-1], cont[ii], p);
            const T currentDistance = (p-currentResult.first).dot(p-currentResult.first);
            if (currentDistance < bestDistance) {
                bestDistance = currentDistance;
                bestProjection = currentResult.first;
                bestStart = cont[ii-1];
                bestEnd = cont[ii];
                bestTime = currentResult.second;
            }
        }
    }
    return std::make_tuple(bestProjection, bestStart, bestEnd, bestTime);
}

template<class P>
P ContourFeatures::pointContourProjection(const P p) const {
    return std::get<0>(pointContourProjectionSub<double>(p));
}

namespace {
template<class T>
int ipow(T base, size_t exp)
{
    T result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}
void handle_integration(double& result, cv::Point const& p, cv::Point const& projection, size_t const norm) {
    auto const diff = projection - p;
    double const lengthsq = diff.x * diff.x + diff.y * diff.y;
    if (0 == norm) {
        result = std::max(result, lengthsq);
    }
    else if (1 == norm % 2) {
        double const length = std::sqrt(lengthsq);
        result += ipow(lengthsq, norm/2) * length;
    }
    else {
        result += ipow(lengthsq, norm/2);
    }
}
}

double ContourFeatures::meanDistance(const ContourFeatures &other, size_t const norm) {
    double result = 0;
    size_t counter = 0;
    size_t num_points = countNodes() + other.countNodes();
    for (auto const& p : getPoints(num_points)) {
        auto const projection = other.pointContourProjection(p);
        handle_integration(result, p, projection, norm);
        counter++;
    }
    for (auto const& p : other.getPoints(num_points)) {
        auto const projection = pointContourProjection(p);
        handle_integration(result, p, projection, norm);
        counter++;
    }
    if (0 == norm) {
        return result;
    }
    return result / counter;
}

double ContourFeatures::meanDistance(const ContourFeatures &other, cv::Mat_<double> const flow, size_t const norm) {
    ContourFeatures clone(other);
    clone.applyMatrix(flow);
    return meanDistance(clone, norm);
}

cv::Point2d ContourFeatures::applyMatrix(const cv::Mat_<double> &mat, const cv::Point2d point) {
    unsigned int const cols = mat.cols;
    unsigned int const rows = 1;
    cv::Mat_<double> src(cols, rows);
    src(0,0) = point.x;
    src(1,0) = point.y;
    src(2,0) = 1.0;
    src = mat * src;
    cv::Point2d result(src(0,0), src(1,0));
    if (3 == mat.cols && 3 == mat.rows) {
        result /= src(2);
    }
    return result;
}

void ContourFeatures::applyMatrix(const cv::Mat_<double> &mat) {
    resetCache();
    for (auto &contour : areaData) {
        for (auto &point: contour) {
            point = applyMatrix(mat, point);
        }
    }
    for (auto &contour : lineData) {
        for (auto &point: contour) {
            point = applyMatrix(mat, point);
        }
    }
}

void ContourFeatures::drawArea(
        cv::Mat& image,
        cv::Scalar color,
        cv::Point2d offset) const {

    for (const auto& contour : areaData) {
        const size_t currentSize = contour.size();
        cv::Point * intData = new cv::Point[currentSize];
        for (size_t jj = 0; jj < currentSize; ++jj) {
            intData[jj] = cv::Point(static_cast<int>(std::round(contour[jj].x + offset.x)),
                                    static_cast<int>(std::round(contour[jj].y + offset.y)));
            if (intData[jj].x < 0 || intData[jj].x < 0 || intData[jj].x >= image.cols || intData[jj].y >= image.rows) {
                std::cerr << "out of range in drawArea:" << intData[jj].x << ", " << intData[jj].y << std::endl;
            }
        }
        const cv::Point* data[1] = { intData };
        int numPoints[1] = {static_cast<int>(currentSize)};
        cv::fillPoly(image, data, numPoints, 1, color);
        delete [] intData;
    }
}

void ContourFeatures::removeOutsideLines(size_t width, size_t height) {
    double const offset = 10000;
    ContourFeatures ring;
    ring.areaData.resize(2);
    auto& inner_ring = ring.areaData.front();
    auto& outer_ring = ring.areaData.back();
    inner_ring.reserve(4);
    outer_ring.reserve(4);
    inner_ring.push_back(cv::Point2d(0,0));
    inner_ring.push_back(cv::Point2d(width,0));
    inner_ring.push_back(cv::Point2d(width,height));
    inner_ring.push_back(cv::Point2d(0,height));
    //inner_ring.push_back(cv::Point2d(0,0));

    outer_ring.push_back(cv::Point2d(-offset,-offset));
    outer_ring.push_back(cv::Point2d(-offset,height+offset));
    outer_ring.push_back(cv::Point2d(width + offset,height+offset));
    outer_ring.push_back(cv::Point2d(width + offset,-offset));
    //outer_ring.push_back(cv::Point2d(-offset,-offset));

    removeOverlappedLines(ring);
}

void ContourFeatures::removeOverlappedLines(const ContourFeatures &other) {
    Geom::PathVector self_path = getLinesPathVector();
    Geom::PathVector other_path = other.getAreaPathVector();
    //self_path = self_path.removeLineOverlap(other_path);
    throw std::runtime_error("Need to fix lib2geom before using this function");
    setLines(self_path);
}



void ContourFeatures::calculateMinMax() const {
    if (hasMinMax) {
        return;
    }
    hasMinMax = true;
    min = max = areaData.front().front();
    for (auto contour : areaData) {
        for (auto e: contour) {
            min = pmin(min, e);
            max = pmax(max, e);
        }
    }
}


Geom::PathVector ContourFeatures::getAreaPathVector() const {
    Geom::PathVector result;
    for (const auto& area : areaData) {
        Geom::Path current_path(Geom::Point(area.front().x, area.front().y));
        for (size_t ii = 1; ii < area.size(); ++ii) {
            current_path.appendNew<Geom::LineSegment>(Geom::Point(area[ii].x, area[ii].y));
        }
        current_path.close();
        result.push_back(current_path);
    }
    return result;
}

Geom::PathVector ContourFeatures::getLinesPathVector() const {
    Geom::PathVector result;
    for (const auto& line : lineData) {
        if (!line.empty()) {
            Geom::Path current_path(Geom::Point(line.front().x, line.front().y));
            for (size_t ii = 1; ii < line.size(); ++ii) {
                /*
                Geom::Point p1(line[ii-1].x, line[ii-1].y);
                Geom::Point p2(line[ii].x, line[ii].y);
                current_path.append(Geom::LineSegment(p1, p2));
                */
                current_path.appendNew<Geom::LineSegment>(Geom::Point(line[ii].x, line[ii].y));
            }
            result.push_back(current_path);
        }
    }
    return result;
}

void ContourFeatures::setLines(const Geom::PathVector &path) {
    lineData.clear();
    lineData.reserve(path.size());
    for (auto const& p : path) {
        std::vector<cv::Point2d> current_line;
        current_line.reserve(p.size());
        current_line.push_back(cv::Point2d(p.front().initialPoint().x(), p.front().initialPoint().y()));
        for (auto const& line : p) {
            current_line.push_back(cv::Point2d(line.finalPoint().x(), line.finalPoint().y()));
        }
        lineData.push_back(current_line);
    }
}

size_t ContourFeatures::nodeCount() const {
    size_t result = 0;
    for (const auto& line : lineData) {
        result += line.size();
    }
    return result;
}



double ContourFeatures::disparityUncertainty(const double estimated_disparity) const {

    // Estimated distance between cardboard and focal point:
    double const estimated_distance = std::abs(FB / estimated_disparity);

    // Minimum true disparity
    double const dmin = std::abs(FB / (estimated_distance + unc_meter));

    // Maximum true disparity
    double const dmax = FB / (estimated_distance - unc_meter);

    double uncertainty = 0;
    if (dmax < 0 || !std::isfinite(dmax)) {
        uncertainty = std::abs(std::abs(estimated_disparity) - dmin);
        //std::cout << "A ";
    }
    else {
        uncertainty = std::max(
                    std::abs(std::abs(estimated_disparity) - dmin),
                    std::abs(std::abs(estimated_disparity) - dmax)
                    );
        //std::cout << "B ";
    }
    uncertainty = std::max(1.0, uncertainty);

    //std::cout << "Type: " << type << ", disp: " << estimated_disparity << ", dist: " << estimated_distance << ", unc: "
    //          << uncertainty << " unc_meter: " << unc_meter
    //          << " relative: " << std::abs(uncertainty / estimated_disparity) << std::endl;

    return uncertainty;

}

void ContourFeatures::close() {
    for (auto& line : lineData) {
        if (line.size() > 1) {
            if (dist(line.back(), line.front()) > 1e-16) {
                line.push_back(line.front());
            }
        }
    }
}

ContourFeatures::ContourFeatures(const ContourFeatures &orig) : areaData(orig.areaData), lineData(orig.lineData) {

}

cv::Point2d ContourFeatures::areaMedian() const {
    calculateMinMax();
    cv::Point2d dim = max - min + cv::Point2d(2,2);
    cv::Point2d offset = min - cv::Point2d(1,1);
    cv::Mat_<uint8_t> img(
                static_cast<size_t>(std::ceil(dim.y)),
                static_cast<size_t>(std::ceil(dim.x)),
                static_cast<uint8_t>(0));
    drawArea(img, cv::Scalar(255), -offset);
    std::vector<uint16_t> x, y;
    x.reserve(img.cols * img.rows);
    y.reserve(img.cols * img.rows);
    for (int yy = 0; yy < img.rows; ++yy) {
        uint8_t* imgRow = img.ptr<uint8_t>(yy);
        for (int xx = 0; xx < img.cols; ++xx) {
            if (imgRow[xx] > 0) {
                x.push_back(xx);
                y.push_back(yy);
            }
        }
    }
    if (x.empty() || y.empty()) {
        return cv::Point2d(0,0);
    }
    size_t const mid_index = x.size()/2;
    std::nth_element(x.begin(), x.begin() + mid_index, x.end());
    std::nth_element(y.begin(), y.begin() + mid_index, y.end());
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    return cv::Point2d(
                offset.x + *(x.begin() + mid_index),
                offset.y + *(y.begin() + mid_index)
                );
}

cv::Vec2d ContourFeatures::maxDist(const ContourFeatures& b) {
    cv::Point2d maxErr(0,0);
    if (b.getLines().empty() || b.getLines().front().empty()) {
        return maxErr;
    }
    for (const auto& selfContour : lineData) {
        if (selfContour.empty()) {
            continue;
        }
        for (const auto& selfPoint : selfContour) {
            cv::Point2d localMin = abs(b.getLines().front().front() - selfPoint);
            for (const auto& targetContour : b.getLines()) {
                for (const auto& targetPoint : targetContour) {
                    localMin = pmin(localMin, abs(selfPoint - targetPoint));
                }
            }
            maxErr = pmax(maxErr, localMin);
        }
    }
    return cv::Vec2d(maxErr.x, maxErr.y);
}

template<class T>
void ContourFeatures::plotFlow(std::ostream& out, cv::Mat_<T>& mat) {
    for (const auto & contour : lineData) {
        for (const auto & src: contour) {
            const auto target = applyMatrix(mat, src);
            out << src.x << " " << src.y << " " << target.x - src.x << " " << target.y - src.y << std::endl;
        }
    }
}

template<class T>
void ContourFeatures::plotFlow(const std::string filename, cv::Mat_<T>& mat) {
    std::ofstream out(filename);
    plotFlow(out, mat);
    out.close();
}


void ContourFeatures::insert(const cv::Point2d elem) {
    if (areaData.empty()) {
        areaData.resize(1);
    }
    if (lineData.empty()) {
        lineData.resize(1);
    }
    areaData.back().push_back(elem);
    lineData.back().push_back(elem);
    resetCache();
}

void ContourFeatures::insert(Geom::Point const elem) {
    insert(cv::Point2d(elem.x(), elem.y()));
}

void ContourFeatures::insert(const double x, const double y) {
    insert(cv::Point2d(x,y));
}

bool ContourFeatures::visible() const {
    if (areaData.empty() || lineData.empty()) {
        return false;
    }
    bool has_area = false;
    for (auto const& d : areaData) {
        if (!d.empty()) {
            has_area = true;
            break;
        }
    }
    if (!has_area) {
        return false;
    }
    for (auto const& d : lineData) {
        if (!d.empty()) {
            return true;
        }
    }
    return false;
}

void ContourFeatures::applyShift(const cv::Point2d shift) {
    hasCentroid = false;
    hasMinMax = false;
    for (auto &contour : areaData) {
        for (auto &point : contour) {
            point += shift;
        }
    }
    for (auto &contour : lineData) {
        for (auto &point : contour) {
            point += shift;
        }
    }
}

namespace {
template<class T>
size_t count(std::vector<T> const& data) {
    size_t result = 0;
    for (auto const& sub : data) {
        result += sub.size();
    }
    return result;
}

size_t max_size_t(size_t const a, size_t const b) {
    return (a > b) ? a : b;
}
}

size_t ContourFeatures::countNodes() const {
    return max_size_t(count(areaData), count(lineData));
}

//template void ContourFeatures::applyMatrix(cv::Mat_<float> const& mat);
//template void ContourFeatures::applyMatrix(cv::Mat_<double> const& mat);

template void ContourFeatures::plotFlow(const std::string filename, cv::Mat_<float>& mat);
template void ContourFeatures::plotFlow(const std::string filename, cv::Mat_<double>& mat);

template void ContourFeatures::plotFlow(std::ostream& out, cv::Mat_<float>& mat);
template void ContourFeatures::plotFlow(std::ostream& out, cv::Mat_<double>& mat);

void ContourFeatures::dummy_instantiation() const
{
    pointContourProjection(cv::Point2d());
    pointContourProjection(cv::Point_<double>());
    throw std::runtime_error("You shall not pass");
}

template cv::Point_<double> ContourFeatures::pointContourProjection(cv::Point_<double>) const;
