/*
  I/O interface class for loading, storing and manipulating
  optical flow fields in KITTI format. This file requires libpng
  and libpng++ to be installed (for accessing png files). More
  detailed format specifications can be found in the readme.txt

  (c) Andreas Geiger
*/
#include "io_flow.h"

namespace KITTI {

// default constructor
FlowImage::FlowImage () {
    data_   = 0;
    width_  = 0;
    height_ = 0;
}

// construct flow image from png file
FlowImage::FlowImage (const std::string file_name) {
    png::image< png::rgb_pixel_16 > image(file_name);
    readFlowField(image);
}

// copy constructor
FlowImage::FlowImage (const FlowImage &F) {
    width_  = F.width_;
    height_ = F.height_;
    data_   = (float*)malloc(width_*height_*3*sizeof(float));
    memcpy(data_,F.data_,width_*height_*3*sizeof(float));
}

// construct flow field from data
FlowImage::FlowImage (const float* data, const int32_t width, const int32_t height) : width_(width), height_(height) {
    data_ = (float*)malloc(width*height*3*sizeof(float));
    memcpy(data_,data,width*height*3*sizeof(float));
}

// construct empty (= all pixels invalid) flow field of given width / height
FlowImage::FlowImage (const int32_t width, const int32_t height) : width_(width), height_(height) {
    data_ = (float*)malloc(width*height*3*sizeof(float));
    for (int32_t i=0; i<width*height*3; i++)
        data_[i] = 0;
}

FlowImage::FlowImage (cv::Mat const& src) {
    width_ = src.cols;
    height_ = src.rows;
    data_ = (float*)malloc(width_ * height_ * 3 * sizeof(float));

    // u - width - column index
    // v - height - row index

    int32_t u_column_index = 0;
    int32_t v_row_index = 0;

    for (v_row_index = 0; v_row_index < height_; v_row_index++) {
        cv::Vec2f const* flow_row = src.ptr<cv::Vec2f>(v_row_index);
        for (u_column_index = 0; u_column_index < width_; u_column_index++) {
            setFlowU(u_column_index, v_row_index, flow_row[u_column_index][0]);
            setFlowV(u_column_index, v_row_index, flow_row[u_column_index][1]);
            if (!std::isfinite(flow_row[u_column_index][0]) || !std::isfinite(flow_row[u_column_index][1])) {
                setValid(u_column_index, v_row_index, false);
            }
            else {
                setValid(u_column_index, v_row_index, true);
            }
        }
    }
}

cv::Mat FlowImage::getCVMat() const {
    cv::Mat result(height_, width_, CV_32FC2);
    // u - width - column index
    // v - height - row index

    int32_t u_column_index = 0;
    int32_t v_row_index = 0;

    for (v_row_index = 0; v_row_index < height_; v_row_index++) {
        cv::Vec2f * flow_row = result.ptr<cv::Vec2f>(v_row_index);
        for (u_column_index = 0; u_column_index < width_; u_column_index++) {
            if (isValid(u_column_index, v_row_index)) {
                flow_row[u_column_index][0] = getFlowU(u_column_index, v_row_index);
                flow_row[u_column_index][1] = getFlowV(u_column_index, v_row_index);
            }
            else {
                flow_row[u_column_index][0] = std::numeric_limits<float>::quiet_NaN();
                flow_row[u_column_index][1] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    return result;
}

// deconstructor
FlowImage::~FlowImage () {
    if (data_) { free(data_); data_ = 0; }
}

// assignment operator, copies contents of F
FlowImage& FlowImage::operator= (const FlowImage &F) {
    if (this!=&F) {
        if (F.width_!=width_ || F.height_!=height_) {
            free(data_);
            width_  = F.width_;
            height_ = F.height_;
            data_   = (float*)malloc(width_*height_*3*sizeof(float));
        }
        memcpy(data_,F.data_,width_*height_*3*sizeof(float));
    }
    return *this;
}

// read flow field from png file
void FlowImage::read (const std::string file_name) {
    if (data_) { free(data_); data_ = 0; }
    png::image< png::rgb_pixel_16 > image(file_name);
    readFlowField(image);
}


// read flow field from png file
void FlowImage::read (std::istream & in) {
    if (data_) { free(data_); data_ = 0; }
    png::image< png::rgb_pixel_16 > image(in);
    readFlowField(image);
}

void FlowImage::write(png::image<png::rgb_pixel_16> &image) {
    image.resize(width_, height_);
    for (int32_t v=0; v<height_; v++) {
        for (int32_t u=0; u<width_; u++) {
            png::rgb_pixel_16 val;
            val.red   = 0;
            val.green = 0;
            val.blue  = 0;
            if (isValid(u,v)) {
                // Doing calculations in double precision reduces rounding errors.
                // Maximum observed round-trip difference with single precision is 0.00784302
                // with double precision 0.0078125 which is almost .4% less.
                // Also, rounding decreases differences by 50%.
                val.red   = (uint16_t)std::max(std::min(std::round(getFlowU(u,v)*64.0d+32768.0d),65535.0d),0.0d);
                val.green = (uint16_t)std::max(std::min(std::round(getFlowV(u,v)*64.0d+32768.0d),65535.0d),0.0d);
                //val.red   = (uint16_t)std::max(std::min(std::round(getFlowU(u,v)*64.0f+32768.0f),65535.0f),0.0f);
                //val.green = (uint16_t)std::max(std::min(std::round(getFlowV(u,v)*64.0f+32768.0f),65535.0f),0.0f);
                val.blue  = 1;
            }
            image.set_pixel(u,v,val);
        }
    }
}

// write flow field to png file
void FlowImage::write (const std::string file_name) {
    png::image< png::rgb_pixel_16 > image(width_,height_);
    write(image);
    image.write(file_name);
}

// write flow field to png file
void FlowImage::write (std::ostream &out) {
    png::image< png::rgb_pixel_16 > image(width_,height_);
    write(image);
    image.write_stream(out);
}

// write flow field to false color map using the Middlebury colormap
void FlowImage::writeColor (const std::string file_name,float max_flow) {
    if (max_flow<=1.0f)
        max_flow = std::max(maxFlow(),1.0f);
    writeFalseColors (file_name,max_flow);
}

// get optical flow u-component at given pixel
float FlowImage::getFlowU (const int32_t u,const int32_t v) const {
    return data_[3*(v*width_+u)+0];
}

// get optical flow v-component at given pixel
float FlowImage::getFlowV (const int32_t u,const int32_t v) const {
    return data_[3*(v*width_+u)+1];
}

// check if optical flow at given pixel is valid
bool FlowImage::isValid (const int32_t u,const int32_t v) const {
    return data_[3*(v*width_+u)+2]>0.5;
}

// get optical flow magnitude at given pixel
float FlowImage::getFlowMagnitude (const int32_t u,const int32_t v) {
    float fu = getFlowU(u,v);
    float fv = getFlowV(u,v);
    return std::sqrt(fu*fu+fv*fv);
}

// set optical flow u-component at given pixel
void FlowImage::setFlowU (const int32_t u,const int32_t v,const float val) {
    data_[3*(v*width_+u)+0] = val;
}

// set optical flow v-component at given pixel
void FlowImage::setFlowV (const int32_t u,const int32_t v,const float val) {
    data_[3*(v*width_+u)+1] = val;
}

// set optical flow at given pixel to valid / invalid
void FlowImage::setValid (const int32_t u,const int32_t v,const bool valid) {
    data_[3*(v*width_+u)+2] = valid ? 1 : 0;
}

// get maximal optical flow magnitude
float FlowImage::maxFlow () {
    float max_flow = 0;
    for (int32_t u=0; u<width_; u++)
        for (int32_t v=0; v<height_; v++)
            if (isValid(u,v) && getFlowMagnitude(u,v)>max_flow)
                max_flow = getFlowMagnitude(u,v);
    return max_flow;
}

// interpolate all missing (=invalid) optical flow vectors
void FlowImage::interpolateBackground () {

    // for each row do
    for (int32_t v=0; v<height_; v++) {

        // init counter
        int32_t count = 0;

        // for each pixel do
        for (int32_t u=0; u<width_; u++) {

            // if flow is valid
            if (isValid(u,v)) {

                // at least one pixel requires interpolation
                if (count>=1) {

                    // first and last value for interpolation
                    int32_t u1 = u-count;
                    int32_t u2 = u-1;

                    // set pixel to min flow
                    if (u1>0 && u2<width_-1) {
                        float fu_ipol = std::min(getFlowU(u1-1,v),getFlowU(u2+1,v));
                        float fv_ipol = std::min(getFlowV(u1-1,v),getFlowV(u2+1,v));
                        for (int32_t u_curr=u1; u_curr<=u2; u_curr++) {
                            setFlowU(u_curr,v,fu_ipol);
                            setFlowV(u_curr,v,fv_ipol);
                            setValid(u_curr,v,true);
                        }
                    }
                }

                // reset counter
                count = 0;

                // otherwise increment counter
            } else {
                count++;
            }
        }

        // extrapolate to the left
        for (int32_t u=0; u<width_; u++) {
            if (isValid(u,v)) {
                for (int32_t u2=0; u2<u; u2++) {
                    setFlowU(u2,v,getFlowU(u,v));
                    setFlowV(u2,v,getFlowV(u,v));
                    setValid(u2,v,true);
                }
                break;
            }
        }

        // extrapolate to the right
        for (int32_t u=width_-1; u>=0; u--) {
            if (isValid(u,v)) {
                for (int32_t u2=u+1; u2<=width_-1; u2++) {
                    setFlowU(u2,v,getFlowU(u,v));
                    setFlowV(u2,v,getFlowV(u,v));
                    setValid(u2,v,true);
                }
                break;
            }
        }
    }

    // for each column do
    for (int32_t u=0; u<width_; u++) {

        // extrapolate to the top
        for (int32_t v=0; v<height_; v++) {
            if (isValid(u,v)) {
                for (int32_t v2=0; v2<v; v2++) {
                    setFlowU(u,v2,getFlowU(u,v));
                    setFlowV(u,v2,getFlowV(u,v));
                    setValid(u,v2,true);
                }
                break;
            }
        }

        // extrapolate to the bottom
        for (int32_t v=height_-1; v>=0; v--) {
            if (isValid(u,v)) {
                for (int32_t v2=v+1; v2<=height_-1; v2++) {
                    setFlowU(u,v2,getFlowU(u,v));
                    setFlowV(u,v2,getFlowV(u,v));
                    setValid(u,v2,true);
                }
                break;
            }
        }
    }
}

// compute error map of flow field, given the non-occluded and occluded
// ground truth optical flow maps. stores result as color png image.
png::image<png::rgb_pixel> FlowImage::errorImage (FlowImage &F_noc,FlowImage &F_occ,bool log_colors) {
    png::image<png::rgb_pixel> image(width(),height());
    for (int32_t v=1; v<height()-1; v++) {
        for (int32_t u=1; u<width()-1; u++) {
            if (F_occ.isValid(u,v)) {
                png::rgb_pixel val;
                if (log_colors) {
                    float dfu = getFlowU(u,v)-F_occ.getFlowU(u,v);
                    float dfv = getFlowV(u,v)-F_occ.getFlowV(u,v);
                    float f_err = sqrt(dfu*dfu+dfv*dfv);
                    float f_mag = F_occ.getFlowMagnitude(u,v);
                    float n_err = std::min(f_err/3.0,20.0*f_err/f_mag);
                    for (int32_t i=0; i<10; i++) {
                        if (n_err>=LC[i][0] && n_err<LC[i][1]) {
                            val.red   = (uint8_t)LC[i][2];
                            val.green = (uint8_t)LC[i][3];
                            val.blue  = (uint8_t)LC[i][4];
                        }
                    }
                    if (!F_noc.isValid(u,v)) {
                        val.red   *= 0.5;
                        val.green *= 0.5;
                        val.blue  *= 0.5;
                    }
                } else {
                    float dfu = getFlowU(u,v)-F_occ.getFlowU(u,v);
                    float dfv = getFlowV(u,v)-F_occ.getFlowV(u,v);
                    float f_err = std::min(sqrt(dfu*dfu+dfv*dfv),5.0)/5.0;
                    val.red   = (uint8_t)(f_err*255.0);
                    val.green = (uint8_t)(f_err*255.0);
                    val.blue  = (uint8_t)(f_err*255.0);
                    if (!F_noc.isValid(u,v)) {
                        val.green = 0;
                        val.blue  = 0;
                    }
                }
                for (int32_t v2=v-1; v2<=v+1; v2++)
                    for (int32_t u2=u-1; u2<=u+1; u2++)
                        image.set_pixel(u2,v2,val);
            }
        }
    }
    return image;
}

// direct access to private variables
float*  FlowImage::data   () { return data_; }
int32_t FlowImage::width  () const { return width_; }
int32_t FlowImage::height () const { return height_; }

void FlowImage::readFlowField(png::image< png::rgb_pixel_16 > const& image) {
    width_  = image.get_width();
    height_ = image.get_height();
    data_   = (float*)malloc(width_*height_*3*sizeof(float));
    for (int32_t v=0; v<height_; v++) {
        for (int32_t u=0; u<width_; u++) {
            png::rgb_pixel_16 val = image.get_pixel(u,v);
            if (val.blue>0) {
                setFlowU(u,v,((float)val.red  -32768.0f)/64.0f);
                setFlowV(u,v,((float)val.green-32768.0f)/64.0f);
                setValid(u,v,true);
            } else {
                setFlowU(u,v,0);
                setFlowV(u,v,0);
                setValid(u,v,false);
            }
        }
    }
}

float FlowImage::hsvToRgb (float h, float s, float v, float &r, float &g, float &b) {
    float c  = v*s;
    float h2 = 6.0*h;
    float x  = c*(1.0-fabs(fmod(h2,2.0)-1.0));
    if (0<=h2&&h2<1)       { r = c; g = x; b = 0; }
    else if (1<=h2&&h2<2)  { r = x; g = c; b = 0; }
    else if (2<=h2&&h2<3)  { r = 0; g = c; b = x; }
    else if (3<=h2&&h2<4)  { r = 0; g = x; b = c; }
    else if (4<=h2&&h2<5)  { r = x; g = 0; b = c; }
    else if (5<=h2&&h2<=6) { r = c; g = 0; b = x; }
    else if (h2>6) { r = 1; g = 0; b = 0; }
    else if (h2<0) { r = 0; g = 1; b = 0; }
}

void FlowImage::writeFalseColors (const std::string file_name, const float max_flow) {
    float n = 8; // multiplier
    png::image< png::rgb_pixel > image(width_,height_);
    for (int32_t v=0; v<height_; v++) {
        for (int32_t u=0; u<width_; u++) {
            float r=0,g=0,b=0;
            if (isValid(u,v)) {
                float mag = getFlowMagnitude(u,v);
                float dir = atan2(getFlowV(u,v),getFlowU(u,v));
                float h   = fmod(dir/(2.0*M_PI)+1.0,1.0);
                float s   = std::min(std::max(mag*n/max_flow,0.0f),1.0f);
                float v   = std::min(std::max(n-s,0.0f),1.0f);
                hsvToRgb(h,s,v,r,g,b);
            }
            image.set_pixel(u,v,png::rgb_pixel(r*255.0f,g*255.0f,b*255.0f));
        }
    }
    image.write(file_name);
}

} // namespace KITTI
