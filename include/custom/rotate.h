#ifndef ROTATE_INCLUDE_ROTATED_H_
#define ROTATE_INCLUDE_ROTATED_H_
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include <iostream>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <regex>
#include <thread>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
#include "ini.h"
#include "SqliteOperator.h"

using json = nlohmann::json;
using namespace nvinfer1;

enum class CheckCode{
    SUCCEED,
    FASTENER_FAILED,
    SLEEPER_FAILED,
    SQL_FAILED,
    DJI_IMG_FAILED,
    SLEEPER_DET_FAILED,
    FASTENER_DET_FAILED,
    FIRST_CSV_FAILED,
    SECOND_CSV_FAILED,
    FIRST_JSON_FAILED,
    FIRST_IMG_FAILED,
    SECOND_IMG_FAILED

};

struct IniParams{
    std::string fastener_path;
    std::string sleeper_path;
    std::string sql_path;
    std::string dji_img_path ;
    std::string sleeper_detection_path ;
    std::string fastener_detection_path ;
    std::string cropped_img_path ;
    float confidence_threshold ;
    float nms_threshold ;
};

struct StartParams{
    std::string first_csv_path ;
    std::string second_csv_path ;
    std::string first_json_path ;
    std::string first_img_path ;
    std::string second_img_path ;
    std::string save_second_img_path;
    int taskID;
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
    int type; //0:abnormal, 1:normal;
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
    Rotation();
    ~Rotation();
    std::atomic<int> status_{2};//0:initiated, 1:running, 2:stopped, 3:completed;
    std::unique_ptr<CSqliteOperator> csql_;
    std::mutex mtx_;
    std::atomic<bool> break_flag_{false};
    std::atomic<int> total_nums_{0};
    std::atomic<int> current_idx_{0};
    std::atomic<int> current_taskID_{-1};
    CheckCode start(StartParams & start_params);
    CheckCode init(IniParams & ini_params);
    bool check(std::string & path);
    void end();
    int getStatus();
    float getProcess();
    int getTaskID();
    void testMatch();
    void testcudaORB(cv::Mat & img_object, cv::Mat & img_scene);
    cv::Mat cudaORB2(cv::Mat & img_object, cv::Mat & img_scene, cv::Mat & mask);

    std::vector<cv::Point2f> ORB2(cv::Mat & img_object, cv::Mat & img_scene);
    cv::cuda::GpuMat keyPoints2GpuMat(cv::Mat & img);

    cv::Mat test_mat_;
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
    std::vector<SingleResult> runSky(int taskID = -1);
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
    float resize_ratio_ = 0.3;
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
