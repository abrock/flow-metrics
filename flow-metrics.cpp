#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/optflow.hpp>
#include "flowgt.h"
#include <stack>
#include <vector>
#include <map>
#include <ParallelTime/paralleltime.h>
#include "metric-helpers.h"
#include "flowmetrics.h"

#include <tclap/CmdLine.h>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

int main(int argc, char** argv) {
    ParallelTime totalRuntime;

    std::string gtName = "";
    std::string uncName = "";

    bool debugSwitch = false;

    std::vector<std::string> algoResults;

    bool save_all_images = false;

    try {

        // Define the command line object, and insert a message
        // that describes the program. The "Command description message"
        // is printed last in the help text. The second argument is the
        // delimiter (usually space) and the last one is the version number.
        // The CmdLine object parses the argv array based on the Arg objects
        // that it contains.
        TCLAP::CmdLine cmd("flow-metrics takes one ground truth file and any number of algorithm result files and computes all flow metrics", ' ', "0.9");

        // Define a value argument and add it to the command line.
        // A value arg defines a flag and a type of value that it expects,
        // such as "-n Bishop".
        TCLAP::ValueArg<std::string> gtArg("g","gt","filename of the ground truth file",true,"","string");

        // Add the argument nameArg to the CmdLine object. The CmdLine object
        // uses this Arg to parse the command line.
        cmd.add( gtArg );

        TCLAP::ValueArg<std::string> uncArg("u","unc","filename of the uncertainty file for the ground truth",false,"","string");
        cmd.add( uncArg );

        TCLAP::MultiArg<std::string> algoArg("a","algo","filename(s) of the algorithm result file(s)",true,"string");
        cmd.add( algoArg );

        TCLAP::SwitchArg saveArg("s","save","save debugging images of the metric computation process", cmd, false);

        // Define a switch and add it to the command line.
        // A switch arg is a boolean argument and only defines a flag that
        // indicates true or false.  In this example the SwitchArg adds itself
        // to the CmdLine object as part of the constructor.  This eliminates
        // the need to call the cmd.add() method.  All args have support in
        // their constructors to add themselves directly to the CmdLine object.
        // It doesn't matter which idiom you choose, they accomplish the same thing.
        TCLAP::SwitchArg debugSwitchArg("d","debug","Show the debugging window which lets the user explore stuff", cmd, false);

        // Parse the argv array.
        cmd.parse( argc, argv );

        // Get the value parsed by each arg.
        gtName = gtArg.getValue();
        debugSwitch = debugSwitchArg.getValue();
        uncName = uncArg.getValue();
        algoResults = algoArg.getValue();
        save_all_images = saveArg.getValue();

    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return 0;
    }

    if (debugSwitch) {
        cv::Mat algo = GT::readOpticalFlow(algoResults[0]);

        cv::Mat gt = GT::readOpticalFlow(gtName);

        cv::Mat unc;
        if (uncName.empty()) {
            unc = cv::Mat(gt.rows, gt.cols, CV_32FC2, cv::Vec2f(.1, .1));
        }
        else {
            unc = cv::readOpticalFlow(uncName);
        }
        FlowMetrics exp;
        exp.safeAllImages();
        exp.algo = algo;

        ParallelTime t;
        exp.init(gt, unc);
        std::string const init_time = t.print();

        t.start();
        exp.prepare_images();
        std::string const prepare_images_time = t.print();

        //*
        t.start();
        //exp.testPlane2();
        std::string const testPlane_time = t.print();
        //*/

        std::cout << "init: " << init_time << std::endl
                  << "prepare images: " << prepare_images_time << std::endl
                  << "testPlane: " << testPlane_time << std::endl;

        exp.run(gt, unc);
    }

    else {
        runEvaluationSingleGtMultipleAlgos(gtName, uncName, algoResults, save_all_images);
    }

    std::cout << "Total runtime: " << totalRuntime.print() << std::endl;
}

