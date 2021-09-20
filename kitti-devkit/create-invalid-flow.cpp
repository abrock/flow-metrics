#include <iostream>

#include "io_flow.h"

int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <filename>" << std::endl;
        return 1;
    }

    int const width = std::stol(argv[1]);
    int const height = std::stol(argv[2]);

    std::string const filename = argv[3];

    KITTI::FlowImage img(width, height);

    img.write(filename);


    return 0;
}
