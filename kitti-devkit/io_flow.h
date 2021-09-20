/*
  I/O interface class for loading, storing and manipulating
  optical flow fields in KITTI format. This file requires libpng
  and libpng++ to be installed (for accessing png files). More
  detailed format specifications can be found in the readme.txt

  (c) Andreas Geiger
*/

#ifndef IO_FLOW_H
#define IO_FLOW_H

#include <string.h>
#include <stdint.h>
#include <cmath>
#include <png++/png.hpp>
#include <opencv2/imgproc.hpp>
#include "log_colormap.h"

namespace KITTI {

class FlowImage {

public:

    // default constructor
    FlowImage ();

    // construct flow image from png file
    FlowImage (const std::string file_name);

    // copy constructor
    FlowImage (const FlowImage &F);

    // construct flow field from data
    FlowImage (const float* data, const int32_t width, const int32_t height);

    // construct empty (= all pixels invalid) flow field of given width / height
    FlowImage (const int32_t width, const int32_t height);

    FlowImage (cv::Mat const& src);

    // deconstructor
    virtual ~FlowImage ();

    // assignment operator, copies contents of F
    FlowImage& operator= (const FlowImage &F);

    // read flow field from png file
    void read (const std::string file_name);

    void read (std::istream & in);

    // write flow field to png file
    void write (const std::string file_name);

    /**
     * @brief Write flow field to given png datastructure.
     * @param[out] image to which the flow field is written.
     */
    void write(png::image<png::rgb_pixel_16> &image);

    void write(std::ostream& out);

    // write flow field to false color map using the Middlebury colormap
    void writeColor (const std::string file_name,float max_flow=-1.0f);

    // get optical flow u-component at given pixel
    float getFlowU (const int32_t u,const int32_t v) const;

    // get optical flow v-component at given pixel
    float getFlowV (const int32_t u,const int32_t v) const;

    // check if optical flow at given pixel is valid
    bool isValid (const int32_t u,const int32_t v) const;

    // get optical flow magnitude at given pixel
    float getFlowMagnitude (const int32_t u,const int32_t v);

    // set optical flow u-component at given pixel
    void setFlowU (const int32_t u,const int32_t v,const float val);

    /**
     * @brief Get an cv::Mat_<cv::Vec2f> version of the flow image
     * @return
     */
    cv::Mat getCVMat() const;

    // set optical flow v-component at given pixel
    void setFlowV (const int32_t u,const int32_t v,const float val);

    // set optical flow at given pixel to valid / invalid
    void setValid (const int32_t u,const int32_t v,const bool valid);

    // get maximal optical flow magnitude
    float maxFlow ();

    // interpolate all missing (=invalid) optical flow vectors
    void interpolateBackground ();

    // direct access to private variables
    float*  data   ();
    int32_t width  () const;
    int32_t height () const;

    png::image<png::rgb_pixel> errorImage (FlowImage &F_noc,FlowImage &F_occ,bool log_colors=false);

private:

    void readFlowField(png::image< png::rgb_pixel_16 > const& image);

    inline float hsvToRgb (float h, float s, float v, float &r, float &g, float &b);

    void writeFalseColors (const std::string file_name, const float max_flow);

public:

    float  *data_;
    int32_t width_;
    int32_t height_;
};

} // Namespace KITTI

#endif // IO_FLOW_H

