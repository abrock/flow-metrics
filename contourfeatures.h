#ifndef CONTOURFEATURES_H
#define CONTOURFEATURES_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>


#include <2geom/pathvector.h>
#include <2geom/path.h>
#include <2geom/point.h>

#include "kahansum.h"

class ContourFeatures {
private:
    mutable double valArea;
    mutable bool hasArea = false;

    mutable double valLineLength;

    mutable cv::Point2d valLineCentroid;

    mutable cv::Point2d valCentroid;
    mutable bool hasCentroid = false;

    std::vector<std::vector<cv::Point2d> > areaData;
    std::vector<std::vector<cv::Point2d> > lineData;

    void resetCache() {
        hasArea = false;
        hasCentroid = false;
        hasMinMax = false;
    }

    mutable cv::Mat_<double> covMat;

    cv::Point2d mutable min, max;
    bool mutable hasMinMax = false;


    template<class P>
    P abs(P val) {
        return P(std::abs(val.x), std::abs(val.y));
    }

    void calculateMinMax() const;

    double momentXXX = 0;
    double momentXXY = 0;
    double momentXYY = 0;
    double momentYYY = 0;

    std::string type;
    double unc_meter = 2;


    std::map<std::string, double> unc_labels = { // uncertainties in meters for each label.
                                                       {"Person", 0.3},
                                                       {"Marker/Pylon", 0.05},
                                                       {"Vehicle", 2.5},
                                                       {"Other Object", 0.1},
                                                       {"Puddle", -1},
                                                       {"Unsure", 2},
                                                       {"Person On Vehicle", 0.3}
                                                     };

    double FB = 0.3 * 1850.0;  // focal length x baseline. yes this is hardcoded for now- but wont have a big effect anyway (brought to you by Rahul Nair)

    void dummy_instantiation() const;

public:


    Geom::PathVector getAreaPathVector() const;

    Geom::PathVector getLinesPathVector() const;

    void setLines(Geom::PathVector const& path);

    ContourFeatures(std::vector<Geom::Point> const& data) {
        append(data.begin(), data.end());
    }


    void insert(Geom::Point const elem);

    size_t nodeCount() const;

    double getUncMeter() const {
        return unc_meter;
    }

    void close();

    ContourFeatures() {}

    ContourFeatures(std::vector<cv::Point2d> const& data) {
        append(data.begin(), data.end());
    }


    ContourFeatures(ContourFeatures const& orig);

    ContourFeatures(std::vector<cv::Point2d> const& data, std::string const& _type) {
        append(data.begin(), data.end());
        setType(_type);
    }

    template<class Time, class Point>
    std::pair<Point, Time> pointLineProjectionSub(const Point v, const Point w, const Point p) const;

    template<class T, class P>
    P pointLineProjection(const P v, const P w, const P p) const;

    template<class T, class P>
    std::tuple<P, P, P, T> pointContourProjectionSub(const P p) const;

    template<class P>
    P pointContourProjection(const P p) const;


    /**
     * @brief disparityUncertainty Estimates the uncertainty of the cardboard disparity by evaluating
     * the type of the object / person and the estimated GT disparity.
     * @param estimated_disparity Estimated cardboard disparity
     * @return estimated undertainty
     */
    double disparityUncertainty(double const estimated_disparity) const;

    void setType(std::string const& t) {
        type = t;
        auto const it = unc_labels.find(type);
        if (it == unc_labels.end()) {
            unc_meter = 2;
        }
        else {
            unc_meter = (*it).second;
        }
    }

    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }


    std::vector<cv::Point2d> getPoints(const size_t numPoints) const;

    std::vector<double> getPointArray(const size_t numPoints) const;

    template<class P>
    static P pmin(P a, P b) {
        return P(std::min(a.x, b.x), std::min(a.y, b.y));
    }

    template<class P>
    static P pmax(P a, P b) {
        return P(std::max(a.x, b.x), std::max(a.y, b.y));
    }

    cv::Vec2d maxDist(const ContourFeatures& b);

    void reverse() {
        for (auto & contour: areaData) {
            std::reverse(contour.begin(), contour.end());
        }
        for (auto & contour : lineData) {
            std::reverse(contour.begin(), contour.end());
        }
        resetCache();
    }

    cv::Point2d areaMedian() const;

    const std::vector<std::vector<cv::Point2d> >& getLines() const {
        return lineData;
    }

    std::vector<std::vector<cv::Point2d> >& getLines() {
        return lineData;
    }

    std::vector<std::vector<cv::Point2d> >& getAreas() {
        return areaData;
    }

    double sgn_sqrt(const double x) {
        return sgn(x) * std::sqrt(sgn(x) * x);
    }

    double sgn_pow(const double x, const double exp) {
        return sgn(x) * std::pow(sgn(x) * x, exp);
    }

    /**
     * @brief size Number of features extracted from the contour.
     * 1x Area,
     * 2x centroid,
     * 3x 2nd order moments,
     * 4x 3rd order moments,
     * 2x Center point x/y
     */
    const static size_t size = 10;



    void getFeatures(cv::Mat_<double>& vector) {
        centroid();
        covarianceMatrix();
        vector(0) = area();
        vector(1) = valCentroid.x;
        vector(2) = valCentroid.y;
#if 0
        vector(3) = sgn_sqrt(covMat(0,0));
        vector(4) = sgn_sqrt(covMat(1,1));
        vector(5) = sgn_sqrt(covMat(0,1));
        vector(6) = sgn_pow(momentXXX, 1.0/3.0);
        vector(7) = sgn_pow(momentXXY, 1.0/3.0);
        vector(8) = sgn_pow(momentXYY, 1.0/3.0);
        vector(9) = sgn_pow(momentYYY, 1.0/3.0);
#else
        vector(3) = covMat(0,0);
        vector(4) = covMat(1,1);
        vector(5) = covMat(0,1);
        vector(6) = momentXXX;
        vector(7) = momentXXY;
        vector(8) = momentXYY;
        vector(9) = momentYYY;
#endif
        /*
        vector(10) = valLineLength;
        vector(11) = valLineCentroid.x;
        vector(12) = valLineCentroid.y;
        */
        /*
        vector(13) = sgn_sqrt(covMat(0,0));
        vector(14) = sgn_sqrt(covMat(1,1));
        vector(15) = sgn_sqrt(covMat(0,1));
        */
        /*
        vector(16) = sgn_pow(momentXXX, 1.0/3.0);
        vector(17) = sgn_pow(momentXXY, 1.0/3.0);
        vector(18) = sgn_pow(momentXYY, 1.0/3.0);
        vector(19) = sgn_pow(momentYYY, 1.0/3.0);
        */

        //vector(10) = getExcentricity();
        //      Meta stats of matching rates: 0.844064 +- 0.043988, 37 Samples, range: [0.740626, 0.916227]
        //         Taking min/max into account hurts detection:
        //        calculateMinMax();
        //        vector(10) = min.x;
        //        vector(11) = min.y;
        //        vector(12) = max.x;
        //        vector(13) = max.y;
        //        Meta stats of matching rates: 0.780249 +- 0.0491673, 37 Samples, range: [0.688112, 0.885764]

    }



    cv::Point2d getMin() const {
        calculateMinMax();
        return min;
    }

    cv::Point2d getMax() const {
        calculateMinMax();
        return max;
    }

    cv::Point2d getCenter() const {
        calculateMinMax();
        return (max+min)/2;
    }

    void insert(const double x, const double y);

    template<class iter>
    void append(const iter& start, const iter& stop) {
        for (iter it =  start; it != stop; ++it) {
            insert(*it);
        }
    }

    bool visible() const;

    void insert(const cv::Point2d elem);


    void applyMatrix(cv::Mat_<double> const& mat);

    void applyShift(const cv::Point2d shift);

    size_t countNodes() const;

    cv::Point2d centroid() const;
    cv::Mat_<double> covarianceMatrix();

    double meanDistance(ContourFeatures const& other, size_t const norm = 1);

    double meanDistance(ContourFeatures const& other, cv::Mat_<double> const flow, size_t const norm = 1);

    double lineLength() const {
        centroid();
        return valLineLength;
    }

    cv::Point2d lineCentroid() const {
        centroid();
        return valLineCentroid;
    }

    void plotContour(std::ostream& out) const;
    void plotContour(const std::string filename) const;

    template<class T>
    void plotFlow(std::ostream& out, cv::Mat_<T>& mat);
    template<class T>
    void plotFlow(const std::string filename, cv::Mat_<T>& mat);

    static cv::Point2d applyMatrix(cv::Mat_<double> const& mat, const cv::Point2d point);

    void printContour(std::ostream& out) {
        for (auto & contour: areaData) {
            for (auto c : contour) {
                out << c << std::endl;
            }
        }
    }

    double getExcentricity() {
        covarianceMatrix();
        return (covMat(0,0) - covMat(1,1) * covMat(0,0) - covMat(1,1) + 4 * covMat(0,1) * covMat(0,1)) / ((covMat(0,0) + covMat(1,1)) * (covMat(0,0) + covMat(1,1)));
    }

    double getMomentXXX() {
        covarianceMatrix();
        return momentXXX;
    }

    double getMomentYYY() {
        covarianceMatrix();
        return momentYYY;
    }

    double getMomentXXY() {
        covarianceMatrix();
        return momentXXY;
    }

    double getMomentXYY() {
        covarianceMatrix();
        return momentXYY;
    }

    cv::Point2d variance() {
        covarianceMatrix();
        return cv::Point2d(covMat(0,0), covMat(1,1));
    }

    double covariance() {
        covarianceMatrix();
        return covMat(0,1);
    }

    double area() {
        centroid();
        return valArea;
    }

    void calcLineCentroid() const;
    void calcAreaCentroid() const;

    void drawArea(
            cv::Mat& image,
            cv::Scalar color,
            cv::Point2d offset = cv::Point2d(0,0)) const;

    void removeOutsideLines(size_t width, size_t height);
    void removeOverlappedLines(ContourFeatures const& other);
};

#endif // CONTOURFEATURES_H
