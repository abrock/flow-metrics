#include "readgt.h"
#include <boost/filesystem.hpp>

template<class T>
template<class C>
void ReadGT<T>::addFiles(C& files) {
    for (auto it: files) {
        addToIndex(it);
    }
}

template<class T>
template<class S>
void ReadGT<T>::readFile(const S& filename) {
#if HDF5_SUPPORT
    std::cout << "Reading file " << filename << std::endl;
    if (!vigra::isHDF5(filename.c_str())) {
        std::cerr << "File " << filename << " is no HDF5 file!" << std::endl;
        return;
    }
    vigra::HDF5File file(filename, vigra::HDF5File::Open);
    file.setReadOnly();
    std::cout << "Attributes: " << file.listAttributes("sequence") << std::endl;
    try {
        MyVec<std::string> frames;

        //if (!file.existsAttribute("sequence", "label")) {
        //     throw StringException("File doesn't contain attribute 'label'");
        //}
        file.cd("sequence");
        //vigra::HDF5Handle label = file.getAttributeHandle("/sequence", "label");
        std::string label;
        file.readAttribute("/sequence", "label", label);
        std::cout << "Label: " << label << std::endl;

        file.cd("frames");
        file.ls(frames);
        std::cout << "Found frames: " << frames.elements << std::endl;
    }
    catch(std::exception e) {
        std::cout << "Reading file " << filename << " failed: " << std::endl << e.what() << std::endl;
        file.close();
        return;
    }
#else
    throw std::runtime_error("No HDF5 support");
#endif
}

template<class T>
void ReadGT<T>::readFlow(vigra::HDF5File& file, const std::string& name, GT& gt, const int offset) {
    file.cd(name);
    file.cd("maps");
    if (0 == offset) {
        file.cd("disp");
    }
    else {
        file.cd("flow");
        file.cd(std::to_string(offset));
    }
    vigra::MultiArray<3, double> v_density, v_flow, v_uncert;
    file.readAndResize("density", v_density);
    file.readAndResize("map", v_flow);
    file.readAndResize("uncertainty", v_uncert);

    if (v_uncert.shape() != v_flow.shape()) {
        throw StringException(std::string("Shapes of density and flow do not match, aborting\nfile: ") + __FILE__ + "\nline: " + std::to_string(__LINE__));
    }

    const int num_rows = v_flow.shape()[2];
    const int num_cols = v_flow.shape()[1];

    gt.density = cv::Mat(num_rows, num_cols, CV_32FC1);
    gt.flow  = cv::Mat(num_rows, num_cols, CV_32FC2, cv::Scalar(0,0));
    gt.unc   = cv::Mat(num_rows, num_cols, CV_32FC2);

    for (int yy = 0; yy < num_rows; ++yy) {
        for (int xx = 0; xx < num_cols; ++xx) {
            gt.flow.at<cv::Vec2f>(yy, xx)[0] = static_cast<float>(v_flow(0,xx,yy));
            gt.flow.at<cv::Vec2f>(yy, xx)[1] = static_cast<float>(v_flow(1,xx,yy));
            gt.unc.at<cv::Vec2f>(yy, xx)[0]  = static_cast<float>(v_uncert(0,xx,yy));
            gt.unc.at<cv::Vec2f>(yy, xx)[1]  = static_cast<float>(v_uncert(1,xx,yy));
            gt.density.at<float>(yy, xx)     = static_cast<float>(v_density(0,xx,yy));
        }
    }

}

template<class T>
bool ReadGT<T>::addToIndex(const std::string& filename) {
    throw(0);
}

template<class T>
bool ReadGT<T>::addDirToIndex(const std::string& dirname) {

    namespace fs = boost::filesystem;
    // list all files in current directory.
    //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
    fs::path p (dirname);

    fs::directory_iterator end_itr;

    // cycle through the directory
    for (fs::directory_iterator itr(p); itr != end_itr; ++itr)
    {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (fs::is_regular_file(itr->path())) {
            // assign current file name to current_file and echo it out to the console.
            std::string current_file = itr->path().string();
            std::cout << current_file << std::endl;
            addToIndex(current_file);
        }
    }
    return true;
}

template<class T>
bool ReadGT<T>::findFlowInFile(const std::string& img_name, const std::string& filename, GT& flow, const int offset) {
    throw(0);
}

template<class T>
void ReadGT<T>::findFlow(const std::string name, GT& flow, const int offset) {
    throw(0);
}

template<class T>
void ReadGT<T>::printIndex(std::ostream& out) const {
    std::map<std::string, std::vector<std::string> > newmap;
    for (auto const& it : frame_index) {
        newmap[it.second].push_back(it.first);
    }
    for (auto const& file : newmap) {
        out << "File " << file.first << " contains " << file.second.size() << " frames:" << std::endl;
        /*
        for (auto const& frame : file.second) {
            out << "\t" << frame << std::endl;
        }
        */
        out << std::endl;
    }
}

template class ReadGT<float>;
template void ReadGT<float>::addFiles(std::vector<std::string>& files);
