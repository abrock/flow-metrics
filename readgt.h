#ifndef READGT_H
#define READGT_H

#include <vigra/multi_array.hxx>
#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/hdf5impex.hxx>

#include <cstdio>
#include <fstream>
#include <map>
#include <vector>
#include <iostream>

#include <ctime>

#include <exception>
#include <sys/stat.h>
#include <unistd.h>
#include <glob.h>

#include <tclap/CmdLine.h>
#include "vectoroutput.h"
#include "stringexception.h"
#include "flowgt.h"
#include "basename.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

template<class E>
class MyVec {
public:
    std::vector<E> elements;
    void insert(const E& element) {
        elements.push_back(element);
    }
};

template<class T>
class ReadGT {

    int FLOW_VALID_COUNT_THRESHOLD = 50;

    /**
     * @brief index An index mapping scenes to hdf5 files
     */
    std::map<std::string, std::string> index;

    /**
     * @brief frame_index An index mapping frames to hdf5 files
     */
    std::map<std::string, std::string> frame_index;

public:

    void printIndex(std::ostream& out) const;

    template<class C>
    void addFiles(C& files);

    template<class S>
    void readFile(const S& filename);

    /**
     * @brief readFlow Read flow / disp ground truth from a HDF5 file
     *
     * @param file An open HDF5 file, may be read-only.
     * @param name The name of the frame which should be read.
     * @param flow The object for storing the flow information.
     * @param offset If this is zero then disparity will be read, otherwise flow for the
     */
    void readFlow(vigra::HDF5File& file, const std::string& name, GT& flow, const int offset);

    /**
     * @brief addToIndex adds a HDF5 file to the index which later tells us which file contains which scene.
     * @param filename
     */
    bool addToIndex(const std::string& filename);

    bool addDirToIndex(const std::string& dirname);

    /**
     * @brief findFlowInFile searches for the ground truth flow/disp corresponding to the given frame name in the given HDF5 file.
     * If it is found its content is stored in the given FlowGT object and the function returns true,
     * otherwise the function returns false and leaves the FlowGT object unchanged.
     * @param img_name Name of the image. File endings (.jpg, .png etc.) will be ignored.
     * @param filename Path of the HDF5 file, relative to the current directory or absolute.
     * @param flow Object for storing the ground truth, if ground truth is found.
     * @param offset Flow offset. If this is zero then the method searches for stereo instead of flow ground truth.
     * @return True if the flow ground truth is found in the given HDF5 file, otherwise false.
     */
    bool findFlowInFile(const std::string& img_name, const std::string& filename, GT& flow, const int offset);

    /**
     * @brief findFlowInFile searches for the ground truth flow/disp corresponding to the given frame name in all HDF5 files known to the class.
     * If it is found its content is stored in the given FlowGT object and the function returns,
     * otherwise the function throws an exception and leaves the FlowGT object unchanged.
     * @param name Name of the image. File endings (.jpg, .png etc.) will be ignored.
     * @param flow Object for storing the ground truth, if ground truth is found.
     * @param offset Flow offset. If this is zero then the method searches for stereo instead of flow ground truth.
     */
    void findFlow(const std::string name, GT& flow, const int offset);

};

#endif // READGT_H
