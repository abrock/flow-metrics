#include <iostream>
#include <vector>
#include "contourfeatures.h"
#include "contourflow.h"
#include "libColorSelect/gnuplot-iostream/gnuplot-iostream.h"
#include "getexample.h"

#if USE_2GEOM
#include <2geom/point.h>
#endif

namespace gnuplotio {
template<typename T>
struct BinfmtSender<cv::Point_<T> > {
    static void send(std::ostream &stream) {
        BinfmtSender<T>::send(stream);
        BinfmtSender<T>::send(stream);
        BinfmtSender<T>::send(stream);
    }
};

template <typename T>
struct BinarySender<cv::Point_<T> > {
    static void send(std::ostream &stream, const cv::Point_<T> &v) {
        BinarySender<T>::send(stream, v.x);
        BinarySender<T>::send(stream, v.y);
    }
};

// We don't use text mode in this demo.  This is just here to show how it would go.
template<typename T>
struct TextSender<cv::Point_<T> > {
    static void send(std::ostream &stream, const cv::Point_<T> &v) {
        TextSender<T>::send(stream, v.x);
        stream << " ";
        TextSender<T>::send(stream, v.y);
        stream << std::endl;
    }
};
}

void plot(std::string const filename, ContourFeatures const& a, ContourFeatures const& b) {
    namespace gp = gnuplotio;
    gp::Gnuplot plot;
    ContourFlow flow(a,b);
    cv::Mat_<double> sol = flow.getFlow();
    ContourFeatures warped(a);
    warped.applyMatrix(sol);

    cv::Mat_<double> best_ig_sol = flow.getBestIG();
    ContourFeatures best_ig_warped(a);
    best_ig_warped.applyMatrix(best_ig_sol);


    plot << "set term svg" << std::endl
         << "set output \"" << filename << ".svg\"" << std::endl
         << "set xlabel 'x'" << std::endl
         << "set ylabel 'y'" << std::endl
         << "set yrange [:] reverse" << std::endl
         << "set key out horiz" << std::endl
         << "plot " << plot.file1d(a.getLines(), filename + "-a") << " w l title 'A',"
         << plot.file1d(b.getLines(), filename + "-b") << " w l title 'B',"
         << plot.file1d(warped.getLines(), filename + "-warped") << " w l title 'warped',"
         << plot.file1d(best_ig_warped.getLines(), filename + "-best-ig") << " w l title 'best ig',";

}

int main(void) {
    for (size_t ii = 0; ; ++ii) {
        Example psh1, psh2;
        getExample(psh1, psh2, ii);
        std::cout << "Example size: (" << psh1.pts.size() << ", " << psh2.pts.size() << ")" << std::endl;
        ContourFeatures a(psh1.pts);
        ContourFeatures b(psh2.pts);

        plot(std::string("match-example-") + std::to_string(ii), a, b);
    }

}
