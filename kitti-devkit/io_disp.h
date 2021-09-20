/*
  I/O interface class for loading, storing and manipulating
  disparity maps in KITTI format. This file requires libpng
  and libpng++ to be installed for accessing png files. More
  detailed format specifications can be found in the readme.txt
  
  (c) Andreas Geiger
*/

#ifndef IO_DISPARITY_H
#define IO_DISPARITY_H

#include <string.h>
#include <stdint.h>
#include <cmath>
#include <png++/png.hpp>
#include "log_colormap.h"
#include <opencv2/core.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace KITTI {

class DisparityImage {

public:
  
  // default constructor
  DisparityImage ();
  
  // construct disparity image from png file
  DisparityImage (const std::string file_name);

  // copy constructor
  DisparityImage (const DisparityImage &D);
  
  // construct disparity image from data
  DisparityImage (const float* data, const int32_t width, const int32_t height);
  
  // construct empty (= all pixels invalid) disparity map of given width / height
  DisparityImage (const int32_t width, const int32_t height);
  
  DisparityImage (cv::Mat const& src, bool const use_unc = false);

  // deconstructor
  virtual ~DisparityImage ();

  // assignment operator, copies contents of D
  DisparityImage& operator= (const DisparityImage &D);
  
  // read disparity image from png file
  void read (const std::string file_name);
  
  // write disparity image to png file
  void write (const std::string file_name) const;

  // write disparity image to png file
  void write (const char* file_name) const;

  // write disparity image to png file
  void write (const fs::path file_name) const;

  // write disparity image to (grayscale printable) false color map
  // if max_disp<1, the scaling is determined from the maximum disparity
  void writeColor (const std::string file_name,float max_disp=-1.0f);
  
  // get disparity at given pixel
  float getDisp (const int32_t u,const int32_t v) const;

  // is disparity valid
  bool isValid (const int32_t u,const int32_t v) const;
  
  // set disparity at given pixel
  void setDisp (const int32_t u,const int32_t v,const float val);

  // is disparity at given pixel to invalid
  bool setInvalid (const int32_t u,const int32_t v);

  // get maximal disparity
  float maxDisp ();

  // simple arithmetic operations
  DisparityImage operator+ (const DisparityImage &B);

  DisparityImage operator- (const DisparityImage &B);

  DisparityImage abs ();
  
  // interpolate all missing (=invalid) disparities
  void interpolateBackground ();

  // compute error map of current image, given the non-occluded and occluded
  // ground truth disparity maps. stores result as color png image.
  png::image<png::rgb_pixel> errorImage (DisparityImage &D_noc,DisparityImage &D_occ,bool log_colors=false);
  
  // direct access to private variables
  float*  data   ();
  int32_t width  () const;
  int32_t height () const;


  cv::Mat getCVMat() const;
   
private:
  
  void readDisparityMap (const std::string file_name);

  void writeDisparityMap (const std::string file_name) const;

  void writeFalseColors (const std::string file_name, float max_val);

public:
  
  float  *data_;
  int32_t width_;
  int32_t height_;

};

} // namespace KITTI

#endif // DISPARITY_IMAGE_H

