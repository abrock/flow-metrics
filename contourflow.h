#ifndef CONTOURFLOW_H
#define CONTOURFLOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "contourfeatures.h"

#include <ceres/ceres.h>


#include "randutils.hpp"

class ContourFlow;

template<class T>
class CeresPoint {
public:
    T x;
    T y;
    T dot(CeresPoint<T> p) const {
        return p.x * x + p.y * y;
    }
    /**
     * @brief dotrot Calculate the scalar product of p and rot90(self)
     * @param p
     * @return
     */
    T dotrot(CeresPoint<T> p) const {
        return p.x * y - p.y * x;
    }
    T norml2() {
        return ceres::sqrt(x*x + y*y);
    }

    void rotate(T const angle) {
        T const cos = ceres::cos(angle);
        T const sin = ceres::sin(angle);
        T const new_x = cos * x + sin * y;
        y = - sin * x + cos * y;
        x = new_x;
    }

    CeresPoint<T> rotated(double const angle) {
        CeresPoint<T> result(*this);
        result.rotate(angle);
        return result;
    }

    template<class X, class Y>
    CeresPoint(const X _x, const Y _y) : x(T(_x)), y(T(_y)){}
    template<class X>
    CeresPoint(const X p) : x(p.x), y(p.y){}

    CeresPoint(const cv::Vec2d p) : x(p[0]), y(p[1]) {}
    CeresPoint() : x(0), y(0) {}

    cv::Vec2d getVec2d() const {
        return cv::Vec2d(x, y);
    }

    cv::Point2d getPoint2d() const {
        return cv::Point2d(x, y);
    }


};

#include <random>

template<class FLOW>
class FlowBase {
private:
    std::mt19937 engine{randutils::auto_seed_128{}.base()};

public:
    virtual ~FlowBase();

    FlowBase(FLOW& other);

    FlowBase();

    virtual double getParam(size_t ii) final;

    template<class T>
    CeresPoint<T> apply(CeresPoint<T> const& src) const;

    void plotFlow(cv::Mat_<cv::Vec2f> & flow, cv::Mat_<uint8_t> const& mask) const;

    virtual cv::Point2d apply(cv::Point2d const& src) const final;

    virtual cv::Point2d apply(const double data[], cv::Point2d const& src) const final;

    virtual void addNoise(double const scale = 1);
};

template<class T>
std::ostream& operator << (std::ostream& out, CeresPoint<T> const& p) {
    return out << "(" << p.x << ", " << p.y << ")";
}

template<class T>
CeresPoint<T> operator - (const CeresPoint<T> a, const CeresPoint<T> b) {
    return CeresPoint<T>(a.x - b.x, a.y - b.y);
}

template<class T>
CeresPoint<T> operator + (const CeresPoint<T> a, const CeresPoint<T> b) {
    return CeresPoint<T>(a.x + b.x, a.y + b.y);
}

template<class T, class F>
CeresPoint<T> operator * (const F factor, const CeresPoint<T> p) {
    return CeresPoint<T>(factor * p.x, factor * p.y);
}

template<class T, class F>
CeresPoint<T> operator / (const F factor, const CeresPoint<T> p) {
    return CeresPoint<T>(p.x / factor, p.y / factor);
}



class HomographyFlow;
class RestrictedFlow;
class SimilarityFlow;
class DispFlow;

class ContourFlow
{
private:
    bool hasFlow = false;
    bool hasProjectiveFlow = false;
    bool hasRestrictedFlow = false;
    bool hasSimilarityFlow = false;
    bool hasBestIG = false;
    bool hasUnc = false;
    cv::Vec2f getUncertainty();

    ContourFeatures a;
    ContourFeatures b;
    cv::Mat_<double> valFlow, valProjectiveFlow, valRestrictedFlow, valSimilarityFlow;
    cv::Mat_<double> bestIG;
    cv::Vec2f valUnc;

    size_t simulationWidth  = 20;
    size_t simulationHeight = 20;

    double HEIGHT = 1080;
    double WIDTH = 2560;

    bool isVisible(double const x, double const y) const;
    bool isVisible(cv::Point2d const& p) const;

    bool debugOutput = false;
public:
    static double getAngle(double const cos, double const sin) {
        return std::atan2(sin, cos);
    }

    cv::Mat_<double> getFlow();
    cv::Mat_<double> getCentroidFlow();

    ContourFeatures getA() {return a;}
    ContourFeatures getB() {return b;}

    void drawCorrespondence(std::string filename);

    ContourFlow(){}
    ContourFlow(ContourFeatures _a, ContourFeatures _b) : a(_a), b(_b) {}
    ceres::Solver::Summary summary;

    void drawFlow(
            cv::Mat_<cv::Vec2f>& flowImg,
            cv::Mat_<cv::Vec2f>& uncImg);

    void plotContour(cv::Mat_<double>& img, const double stddev = 1.0);
    double plotAt(const cv::Point2d point, const ContourFeatures& contour, const double stddev = 1.0);

    void plotProjections(std::ostream& out, const size_t numIt = 50);
    void plotProjections(const std::string filename, const size_t numPoints = 50);
    void gnuplotProjections(const std::string filename, const size_t numPoints = 50);
    cv::Point2d getMedianProjectionFlow(const double trim = 0.1);

    cv::Mat_<double> getMedianAreaFlow();

    cv::Mat_<double> getCenterFlow();
    cv::Mat_<double> getTopLeftFlow();
    cv::Mat_<double> getTopRightFlow();
    cv::Mat_<double> getBottomRightFlow();
    cv::Mat_<double> getBottomLeftFlow();

    cv::Point2d getMin();
    cv::Point2d getMax();

    cv::Mat_<double> getICPFlow(
            const double cauchyParam = 1,
            size_t const num_points = 50,
            size_t const max_it = 100);

    cv::Mat_<double> getLibICPRotationFlow(size_t const num_points = 150, double const inlier_dist = 10);
    cv::Mat_<double> getLibICPTranslationFlow(size_t const num_points = 150, double const inlier_dist = 10);

    std::vector<cv::Mat_<double> > getInitialGuesses();

    cv::Mat_<double> getBestIG();

    cv::Mat_<double> getDisp(const cv::Mat &gt_disp);

    cv::Mat_<double> getProjectiveFlow(const double cauchyParam = -1,
                                       size_t const num_points = 50,
                                       size_t const max_it = 100);

    template<class Flow>
    class FlowFitter {
    private:
        size_t num_points = 0;
        size_t num_points_last_solve = 0;
    public:
        std::vector<cv::Point2d> src;
        std::vector<cv::Point2d> dst;
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        FlowFitter(Flow const& ig,
                          std::vector<cv::Point2d> const& src,
                          std::vector<cv::Point2d> const& dst,
                          const double cauchyParam = -1);
        FlowFitter(Flow const& ig);
        FlowFitter();
        Flow result;

        void addPoint(cv::Point2d const &_src,
                      cv::Point2d const &_dst,
                      double const cauchyParam = -1);

        void solve();

        /**
         * @brief conditional_solve Only solves if the number of points has increased by at least 20% since the last solve.
         * @return
         */
        bool conditional_solve();

        size_t get_num_points() const;
    };

    template<class Flow>
    static Flow getHomographyFlowSub(Flow const& ig,
                                     std::vector<cv::Point2d> const& src,
                                     std::vector<cv::Point2d> const& dst,
                                     const double cauchyParam = -1);

    struct ProjectiveICPCost;
    struct ProjectiveICPCostRelaxed;
    struct RestrictedFlowCost;
    struct DispFlowCost;
    struct SimilarityFlowCost;

    template<class Cost = ProjectiveICPCost>
    static cv::Mat_<double> getProjectiveFlowSub(cv::Mat_<double> const& ig,
                                                 std::vector<cv::Point2d> const& src,
                                                 std::vector<cv::Point2d> const& dst,
                                                 const double cauchyParam = -1);
    static void normalize(cv::Mat_<double> &mat);

    cv::Mat_<double> getRestrictedFlow(const double cauchyParam = -1,
                                       size_t const num_points = 50,
                                       size_t const max_it = 100);

    static RestrictedFlow getRestrictedFlowSub(RestrictedFlow const& ig,
                                               std::vector<cv::Point2d> const& src,
                                               std::vector<cv::Point2d> const& dst,
                                               const double cauchyParam = -1);

    static DispFlow getDispFlowSub(DispFlow const& ig,
                                   std::vector<cv::Point2d> const& src,
                                   std::vector<cv::Point2d> const& dst,
                                   const double cauchyParam = -1);

    cv::Mat_<double> getSimilarityFlow(const double cauchyParam = -1,
                                       size_t const num_points = 50,
                                       size_t const max_it = 100);

    static SimilarityFlow getSimilarityFlowSub(SimilarityFlow const& ig,
                                               std::vector<cv::Point2d> const& src,
                                               std::vector<cv::Point2d> const& dst,
                                               const double cauchyParam = -1);


};

class DispFlow : public FlowBase<DispFlow> {
public:
    using FlowBase::apply;
    using FlowBase::getParam;

    static uint8_t const num_params = 3;
    double data[3];
    /*
    double & shift_x = data[0];
    double & proj_x = data[1];
    double & proj_y = data[2];
*/

    double shift_x() const {return data[0];}
    double proj_x()   const {return data[1];}
    double proj_y()   const {return data[2];}

    void shift_x (double const val) {data[0] = val;}
    void proj_x  (double const val) {data[1] = val;}
    void proj_y  (double const val) {data[2] = val;}

    void normalize() {
    }

    DispFlow() {
        shift_x(0);
        proj_x(0);
        proj_y(0);
    }

    DispFlow(cv::Mat_<double> const& _mat) {
        double const input_scale = cv::norm(_mat, cv::NORM_INF) / _mat(2,2);
        if (input_scale > 1e10) {
            throw std::runtime_error("Source matrix badly scaled, can't be normalized for a_22 = 1");
        }
        double & shift_x = data[0];
        double & proj_x = data[1];
        double & proj_y = data[2];


        cv::Mat_<double> mat = _mat.clone();
        mat /= mat(2,2);
        shift_x = mat(0,2);
        proj_x = mat(2,0);
        proj_y = mat(2,1);
    }

    template<class T>
    void param2mat(T const data[], T matrix[]) const;

    template<class T>
    void param2mat(T matrix[]) const;

    template<class T, class U>
    CeresPoint<U> apply(T const data[], CeresPoint<U> const& src) const;

    std::string print() const;

    cv::Mat_<double> getMatrix(double const data[]) const;

    cv::Mat_<double> getMatrix() const;

    /**
     * @brief rescale Adjusts the flow to match the changes introduced by scaling the corresponding images by a given factor.
     * @param factor Scale factor. "2" means that the new image has 2x the width and height of the old image.
     */
    void rescale(double const factor);
};

class RestrictedFlow : public FlowBase<RestrictedFlow> {
public:
    using FlowBase::apply;

    // CeresPoint<U> RestrictedFlow::apply(const T*, const CeresPoint<U>&)’
    // CeresPoint<U> RestrictedFlow::apply(const T data[], const CeresPoint<U> &src) {

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);

    static uint8_t const num_params = 6;
    double data[6];
    /*
    double & rotation = data[0];
    double & scale = data[1];
    double & shift_x = data[2];
    double & shift_y = data[3];
    double & proj_x = data[4];
    double & proj_y = data[5];
*/

    double rotation() const {return data[0];}
    double scale()    const {return data[1];}
    double shift_x()  const {return data[2];}
    double shift_y()  const {return data[3];}
    double proj_x()   const {return data[4];}
    double proj_y()   const {return data[5];}

    void rotation(double const val) {data[0] = val;}
    void scale   (double const val) {data[1] = val;}
    void shift_x (double const val) {data[2] = val;}
    void shift_y (double const val) {data[3] = val;}
    void proj_x  (double const val) {data[4] = val;}
    void proj_y  (double const val) {data[5] = val;}

    void normalize() {
        if (rotation() >= M_PI || rotation() <= -M_PI) {
            rotation(fmod(rotation(), 2*M_PI));
        }
    }

    RestrictedFlow() {
        rotation(0);
        scale(1);
        shift_x(0);
        shift_y(0);
        proj_x(0);
        proj_y(0);
    }

    RestrictedFlow(cv::Mat_<double> const& _mat) {
        double const input_scale = cv::norm(_mat, cv::NORM_INF) / _mat(2,2);
        if (input_scale > 1e10) {
            throw std::runtime_error("Source matrix badly scaled, can't be normalized for a_22 = 1");
        }
        double & rotation = data[0];
        double & scale = data[1];
        double & shift_x = data[2];
        double & shift_y = data[3];
        double & proj_x = data[4];
        double & proj_y = data[5];


        cv::Mat_<double> mat = _mat.clone();
        mat /= mat(2,2);
        shift_x = mat(0,2);
        shift_y = mat(1,2);
        proj_x = mat(2,0);
        proj_y = mat(2,1);
        // The determinant of the top left 2x2 sub-matrix is the scale of the mapping.
        double const det = mat(0,0) * mat(1,1) - mat(0,1) * mat(1,0);
        rotation = ContourFlow::getAngle(mat(0,0) + mat(1,1), mat(0,1) - mat(1,0));
        scale = std::abs(det);
    }

    template<class T>
    void param2mat(T const data[], T matrix[]) const;

    template<class T>
    void param2mat(T matrix[]) const;

    std::string print() const;

    cv::Mat_<double> getMatrix(double const data[]) const;

    cv::Mat_<double> getMatrix() const;

    /**
     * @brief rescale Adjusts the flow to match the changes introduced by scaling the corresponding images by a given factor.
     * @param factor Scale factor. "2" means that the new image has 2x the width and height of the old image.
     */
    void rescale(double const factor);
};

class SimilarityFlow : public FlowBase<SimilarityFlow> {
public:
    using FlowBase::apply;
    static uint8_t const num_params = 4;
    double data[4];
    /*
    double & rotation = data[0];
    double & scale = data[1];
    double & shift_x = data[2];
    double & shift_y = data[3];
*/

    double rotation() const {return data[0];}
    double scale()    const {return data[1];}
    double shift_x()  const {return data[2];}
    double shift_y()  const {return data[3];}

    void rotation(double const val) {data[0] = val;}
    void scale   (double const val) {data[1] = val;}
    void shift_x (double const val) {data[2] = val;}
    void shift_y (double const val) {data[3] = val;}

    void normalize() {
        if (rotation() >= M_PI || rotation() <= -M_PI) {
            rotation(fmod(rotation(), 2*M_PI));
        }
    }

    SimilarityFlow() {
        rotation(0);
        scale(1);
        shift_x(0);
        shift_y(0);
    }

    SimilarityFlow(cv::Mat_<double> const& _mat) {
        double const input_scale = cv::norm(_mat, cv::NORM_INF) / _mat(2,2);
        if (input_scale > 1e10) {
            throw std::runtime_error("Source matrix badly scaled, can't be normalized for a_22 = 1");
        }
        double & rotation = data[0];
        double & scale = data[1];
        double & shift_x = data[2];
        double & shift_y = data[3];


        cv::Mat_<double> mat = _mat.clone();
        mat /= mat(2,2);
        shift_x = mat(0,2);
        shift_y = mat(1,2);
        // The determinant of the top left 2x2 sub-matrix is the scale of the mapping.
        double const det = mat(0,0) * mat(1,1) - mat(0,1) * mat(1,0);
        rotation = ContourFlow::getAngle(mat(0,0) + mat(1,1), mat(0,1) - mat(1,0));
        scale = std::abs(det);
    }

    template<class T>
    void param2mat(T const data[], T matrix[]) const;

    template<class T>
    void param2mat(T matrix[]) const;

    template<class T, class U>
    CeresPoint<T> apply(T const data[], CeresPoint<U> const& src) const;

    cv::Mat_<double> getMatrix(double const data[]) const;

    cv::Mat_<double> getMatrix() const;

    std::string printDiff(SimilarityFlow const& other) const;

    std::string print() const;

    /**
     * @brief rescale Adjusts the flow to match the changes introduced by scaling the corresponding images by a given factor.
     * @param factor Scale factor. "2" means that the new image has 2x the width and height of the old image.
     */
    void rescale(double const factor);

};

/**
 * @brief The Polynome0Flow class represents flow fields where u and v are given by polynomes of degree 0 in
 * pixel coordinates x and y:
 * u = a_0
 * v = a_1
 */
class Polynome0Flow : public FlowBase<Polynome0Flow> {
public:
    using FlowBase::apply;
    static const size_t num_params = 2;
    double data[2];

    Polynome0Flow();

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);
};

/**
 * @brief The Polynome1Flow class represents flow fields where u and v are given by polynomes of degree 1 in
 * pixel coordinates x and y:
 * x+u = a_0 + a_1 x + a_2 y
 * y+v = a_3 + a_4 x + a_5 y
 */
class Polynome1Flow : public FlowBase<Polynome1Flow> {
public:
    using FlowBase::apply;
    double data[6];
    static const size_t num_params = 6;

    Polynome1Flow();

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);
};

/**
 * @brief The Polynome2Flow class represents flow fields where u and v are given by polynomes of degree 2 in
 * pixel coordinates x and y:
 * x+u = a_0 + a_1 x + a_2 y + a_3 xy +  a_4 x² +  a_5 y²
 * y+v = a_6 + a_7 x + a_8 y + a_9 xy + a_10 x² + a_11 y²
 */
class Polynome2Flow : public FlowBase<Polynome2Flow> {
public:
    using FlowBase::apply;
    double data[12];
    static const size_t num_params = 12;

    Polynome2Flow();

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);
};

/**
 * @brief The Polynome3Flow class represents flow fields where u and v are given by polynomes of degree 3 in
 * pixel coordinates x and y:
 * x+u = a_0  + a_1 x  + a_2 y  + a_3 xy  +  a_4 x²  +  a_5 y²  +  a_6 x²y +  a_7 xy² +  a_8 x³ +  a_9 y³
 * y+v = a_10 + a_11 x + a_12 y + a_13 xy +  a_14 x² +  a_15 y² + a_16 x²y + a_17 xy² + a_18 x³ + a_19 y³
 */
class Polynome3Flow : public FlowBase<Polynome3Flow> {
public:
    using FlowBase::apply;
    static const size_t num_params = 20;
    double data[num_params];

    Polynome3Flow();

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);
};

class HomographyFlow : public FlowBase<HomographyFlow> {
public:
    using FlowBase::apply;
    static uint8_t const num_params = 9;
    double data[9];

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);

    double getDet() const {
        return (data[0] * data[4] - data[1] * data[3]) / (data[8] * data[8]);
    }

    void sanitize() {

    }

    double frobeniusNorm() const;

    void normalize() {
        double const norm = frobeniusNorm();
        for (size_t ii = 0; ii < 9; ++ii) {
            data[ii] /= norm;
        }
    }

    void addNoise(double const scale = .1);

    void normalize33() {
        for (size_t ii = 0; ii < 9; ++ii) {
            data[ii] /= data[8];
        }
    }

    HomographyFlow() {
        data[1] = data[2] = data[3] = data[5] = data[6] = data[7] = 0;
        data[0] = data[4] = data[8] = 1;
    }

    HomographyFlow(cv::Mat_<double> const& _mat) {
        data[0] = _mat(0,0);
        data[1] = _mat(0,1);
        data[2] = _mat(0,2);
        data[3] = _mat(1,0);
        data[4] = _mat(1,1);
        data[5] = _mat(1,2);
        data[6] = _mat(2,0);
        data[7] = _mat(2,1);
        data[8] = _mat(2,2);
    }

    template<class T>
    void param2mat(T const data[], T matrix[]) const;

    template<class T>
    void param2mat(T matrix[]) const;



    template<class T>
    CeresPoint<T> apply(CeresPoint<T> const& src) const;

    cv::Vec2d apply(cv::Vec2d const& src) const;

    cv::Mat_<double> getMatrix(double const data[]) const;

    cv::Mat_<double> getMatrix() const;

    std::string printDiff(SimilarityFlow const& other) const;

    std::string print() const;

    /**
     * @brief rescale Adjusts the flow to match the changes introduced by scaling the corresponding images by a given factor.
     * @param factor Scale factor. "2" means that the new image has 2x the width and height of the old image.
     */
    void rescale(double const factor);

};

class ParamHomographyFlow : public FlowBase<ParamHomographyFlow> {
public:
    using FlowBase::apply;
    static uint8_t const num_params = 9;
    /**
     * @brief data
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
    double data[9];

    double getDet() const {
        return data[2] * data[3];
    }

    static cv::Mat_<double> getRotationMatrix(double const angle) {
        double const c = std::cos(angle);
        double const s = std::sin(angle);
        return (cv::Mat_<double>(2,2) << c, s, -s, c);
    }

    double frobeniusNorm() const;

    void normalize() {
        // TODO
    }

    void addNoise(double const scale = .1);

    void normalize33() {
        for (size_t ii = 2; ii < 9; ++ii) {
            data[ii] /= data[8];
        }
        sanitize();
    }

    static double sanitizeAngle(double angle) {
        angle = fmod(angle, 2*M_PI);
        if (angle > M_PI) {
            angle -= 2*M_PI;
        }
        if (angle < - M_PI) {
            angle += 2*M_PI;
        }
        return angle;
    }

    void sanitize() {
        if (data[2] < 0 && data[3] < 0) {
            data[2] = -data[2];
            data[3] = -data[3];
            data[0] -= M_PI;
        }
        data[0] = sanitizeAngle(data[0]);
        data[1] = sanitizeAngle(data[1]);
    }

    ParamHomographyFlow(ParamHomographyFlow const& other)
    {
        for (size_t ii = 0; ii < 9; ++ii) {
            data[ii] = other.data[ii];
        }
    }

    ParamHomographyFlow() {
        data[2] = data[3] = data[8] = 1;
        data[0] = data[1] = data[4] = data[5] = data[6] = data[7] = 0;
    }

    template<class T>
    void param2mat(T const data[], T matrix[]) const;

    template<class T>
    void param2mat(T matrix[]) const;

    template<class T, class U>
    static CeresPoint<T> apply(T const data[], CeresPoint<U> const& src);

    template<class T>
    CeresPoint<T> apply(CeresPoint<T> const& src) const;

    cv::Mat_<double> getMatrix(double const data[]) const;

    cv::Mat_<double> getMatrix() const;

    std::string printDiff(SimilarityFlow const& other) const;

    std::string print() const;

    /**
     * @brief rescale Adjusts the flow to match the changes introduced by scaling the corresponding images by a given factor.
     * @param factor Scale factor. "2" means that the new image has 2x the width and height of the old image.
     */
    void rescale(double const factor);

};

std::ostream& operator << (std::ostream& out, ParamHomographyFlow const& in);
std::ostream& operator << (std::ostream& out, SimilarityFlow const& flow);
std::ostream& operator << (std::ostream& out, RestrictedFlow const& flow);


#endif // CONTOURFLOW_H
