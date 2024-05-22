#ifndef ROTATE_INCLUDE_ROTATED_H_
#define ROTATE_INCLUDE_ROTATED_H_
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <regex>
#include <thread>
#include <experimental/filesystem>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
#include <nlohmann/json.hpp>
#include "ini.h"

using json = nlohmann::json;
using namespace nvinfer1;

struct HyperParams{
   std::string csv_path;
   std::string json_path;
   std::string second_csv_path;
   //float resize_ratio;

   std::string first_img_path;
   std::string second_img_path;
};


struct CsvInfo{
    std::string img_path;
    double longitude;//经度
    double latitude;//纬度
    double altitude;//高度

};

struct Location{
    std::string label;
    std::vector<cv::Point2f> pts;
    int type; //0:abnormal,  1: normal
};


struct SingleResult{
    std::string img_path;
    double longitude;//经度
    double latitude;//纬度
    double altitude;//高度
    bool is_valid;
    std::vector<Location> locations;
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Rotation {
public:
    Rotation(){};
    ~Rotation(){
        delete[] sleeper_cls_in_;
        delete[] sleeper_cls_out_;
        delete[] fastener_cls_in_;
        delete[] fastener_cls_out_;
    }

//ground
    void initDetection();
    yolo::Image cvimg(const cv::Mat & image);
    cv::Mat preprocessImg(cv::Mat & img, const int & input_w, const int & input_h, int & padw, int & padh);
    void letterBox(const cv::Mat & image, cv::Mat & outImage, cv::Vec4d & params, const cv::Size & newShape,
                   bool autoShape = false, bool scaleFill = false, bool scaleUp = true, int stride = 32, const cv::Scalar& color = cv::Scalar(127, 127, 127));
    std::vector<std::string> listFiles(const std::string & directory, const std::string & ext);
    void rotateInference(cv::Mat & img, json & j);
    void runGround();

private:
    std::shared_ptr<yolo::Infer> yolo_;
    Logger logger_;
    std::unique_ptr<IRuntime> runtime_obb_;
    std::unique_ptr<ICudaEngine> engine_obb_;
    std::unique_ptr<IExecutionContext> context_obb_;
    const int OBB_BATCH_SIZE_ = 1;
    const int OBB_CHANNELS_ = 3;
    const int OBB_INPUT_H_ = 1024;
    const int OBB_INPUT_W_ = 1024;
    const int OBB_OUTPUT0_BOXES_ = 21504;
    const int OBB_OUTPUT0_ELEMENT_ = 7;
    const int OBB_CLASSES_ = 2;
    float CONF_THRESHOLD_ = 0.35;
    float NMS_THRESHOLD_ = 0.2;
    const char* images_ = "images";
    const char* output0_ = "output0";
    std::vector<std::string> class_names_rot_{"sleeper_normal","sleeper_abnormal"};
    std::vector<std::string> class_names_det_{"fastener_normal","fastener_abnormal","fastener_stone","fastener_missing","rail_big"};
    std::string obb_path_;
    std::string detection_path_;
    std::string dji_img_path_;
    std::string cropped_img_path_;

//sky
public:
    void initParams(HyperParams & params);
    void loadEngine(const std::string& path, const int & flag);
    cv::Mat ORB(cv::Mat & img_object, cv::Mat & img_scene);
    cv::Mat cudaORB(cv::Mat & img_object, cv::Mat & img_scene);
    void testORB();
    void drawSecond(std::vector<Location> & locations,  cv::Mat & img);
    std::vector<CsvInfo> readCsv(std::string & csv_path);
    CsvInfo findFirstInfo(CsvInfo & second_info);
    std::vector<Location> firstToSecond(cv::Mat & H, CsvInfo & first_info);
    void readJson();
    void readAndPreprocess(cv::Mat & img, cv::Mat & temp, std::string & img_path);
    int doClassisication(cv::Mat & img, std::string & label);
    int sleeperClassification(cv::Mat & img);
    int fastenerClassification(cv::Mat & img);
    void skyInference( std::vector<Location> & location, cv::Mat & img);
    void testSky();
    void drawSingleResult(SingleResult & single_result, cv::Mat & img);
    std::vector<SingleResult> runSky();
    void testRunSky();
    void initGroundSky();

private:
    std::string csv_path_;
    std::string json_path_;
    std::string first_img_path_;
    std::string second_img_path_;
    std::vector<CsvInfo> csvinfo_results_; 
    std::string second_csv_path_;
    std::vector<CsvInfo> second_csvinfo_results_;
    std::string save_prefix_;
    int WIDTH_ = 8192;
    int HEIGHT_ = 5460;
    int nfeatures_  = 1000;
    float resize_ratio_ = 0.2;
    float * sleeper_cls_in_ = new float[1*3*224*512]{};
    float * fastener_cls_in_ = new float[1*3*224*224]{};
    float * sleeper_cls_out_ = new float[1*2]{};
    float * fastener_cls_out_ = new float[1*2]{};
    std::unique_ptr<IRuntime> runtime_sleeper_;
    std::unique_ptr<ICudaEngine> engine_sleeper_;
    std::unique_ptr<IExecutionContext> context_sleeper_;
    std::unique_ptr<IRuntime> runtime_fastener_;
    std::unique_ptr<ICudaEngine> engine_fastener_;
    std::unique_ptr<IExecutionContext> context_fastener_;
    const int BATCH_SIZE_ = 1;
    const int FASTENER_INPUT_W_ = 224;
    const int FASTENER_INPUT_H_ = 224;
    const int SLEEPER_INPUT_W_ = 512;
    const int SLEEPER_INPUT_H_ = 224;
    const int CHANNELS_ = 3;
    const int CLASSES_ = 2;
    const char * INPUT_NAMES_ = "input";
    const char * OUTPUT_NAMES_ = "output";

};

#endif //ROTATE_INCLUDE_ROTATED_H_
