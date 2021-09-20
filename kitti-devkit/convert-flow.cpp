#include <iostream>

#include <opencv2/optflow.hpp>
#include "io_flow.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <infile> {outfile}" << std::endl;
        return 0;
    }
    std::string const infile = argv[1];
    bool source_is_kitti = false;
    if (infile.substr(infile.size()-4) == ".png") {
        source_is_kitti = true;
    }
    std::string outfile;
    if (argc < 3) {
        if (source_is_kitti) {
            outfile = infile + ".flo";
        }
        else {
            outfile = infile + ".png";
        }
    }
    else {
        outfile = argv[2];
    }

    if (source_is_kitti) {
        KITTI::FlowImage data(infile);
        cv::writeOpticalFlow(outfile, data.getCVMat());
    }
    else {
        cv::Mat data = cv::readOpticalFlow(infile);
        KITTI::FlowImage converted(data);
        converted.write(outfile);
    }
}
