#include "rotate.h"

using namespace cv;
using namespace std;

void readAndPreprocess(cv::Mat & img, cv::Mat & temp, std::string & img_path) {

        img = cv::imread(img_path);
        cv::resize(img, temp, cv::Size(),1,1);
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);

}


void testabnormal(std::shared_ptr<Rotation> rot){

    cv::Mat img_object, img_scene;
    cv::Mat obj_temp ,scene_temp;

    std::string path1 = "F:/PLATFORM/tools/sift/test.jpg";
    std::string path2 = "F:/PLATFORM/tools/sift/test2.jpg";

    std::thread t1( readAndPreprocess,  std::ref(img_object), std::ref(obj_temp)  , std::ref(path1));
    std::thread t2( readAndPreprocess,  std::ref(img_scene) , std::ref(scene_temp), std::ref(path2));
    t1.join();
    t2.join();

//    CsvInfo first_info;
//    cv::Mat H = rot->cudaORB(obj_temp, scene_temp);

//    cv::Mat out;
//    cv::warpPerspective(img_object, out, H,img_object.size());
//    cv::imwrite("../test.jpg",out);

//    imageSubtract(img_scene ,img_object);

//    std::vector<Location> locations = rot->firstToSecond(H, first_info);

}





int main(){


    std::shared_ptr<Rotation> rot = std::make_shared<Rotation>();
    //rot->initGroundSky();

    IniParams ini_params;
    ini_params.fastener_path = "../models/fastener.engine";
    ini_params.sleeper_path = "../models/sleeper.engine";
    ini_params.dji_img_path = "../testcsv/testcsv_template";
    ini_params.sleeper_detection_path = "../models/sleeper_obb.engine";
    ini_params.fastener_detection_path = "../models/rail_fastener.engine";
    ini_params.cropped_img_path = "../cropped";
    ini_params.confidence_threshold = 0.35;
    ini_params.nms_threshold = 0.2;
    ini_params.sql_path = "../database/ai.rwedb";

    StartParams start_params;
    start_params.first_csv_path = "../testcsv/testcsv_template/test.csv";
    start_params.second_csv_path = "../testcsv/testcsv_inspection/test.csv";
    start_params.first_json_path = "../testcsv/testcsv_template";
    start_params.first_img_path = "../testcsv/testcsv_template";
    start_params.second_img_path = "../testcsv/testcsv_inspection";
    start_params.save_second_img_path = "../testcsv/visualize_results";
    start_params.taskID = 0;

//    rot->init(ini_params);
//    rot->start(start_params);

    rot->testMatch();


}












