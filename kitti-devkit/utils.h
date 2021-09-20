#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <png++/png.hpp>
#include <stdio.h>
#include <math.h>

namespace KITTI {

bool imageFormat(std::string file_name,png::color_type col,size_t depth,int32_t width,int32_t height);

float statMean(std::vector< std::vector<float> > &errors,int32_t idx);

float statWeightedMean(std::vector< std::vector<float> > &errors,int32_t idx,int32_t idx_num);

float statMin(std::vector< std::vector<float> > &errors,int32_t idx);

float statMax(std::vector< std::vector<float> > &errors,int32_t idx);

} // namespace KITTI

#endif // UTILS_H
