#include "rotate.h"

void Rotation::loadEngine(const std::string& path , const int & flag) {
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    std::ifstream file(path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }

    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger_));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(trtModelStream,size));
    std::unique_ptr<IExecutionContext>context(engine->createExecutionContext());

    switch (flag)
    {
    case 0://枕木detection-OBB
    {
        runtime_obb_ = std::move(runtime);
        engine_obb_  = std::move(engine);
        context_obb_ = std::move(context);
        break;
    }
    case 1://枕木classification
    {
        runtime_sleeper_ = std::move(runtime);
        engine_sleeper_  = std::move(engine);
        context_sleeper_ = std::move(context);
        break;
    }
    case 2://铆钉classification
    {
        runtime_fastener_ = std::move(runtime);
        engine_fastener_  = std::move(engine);
        context_fastener_ = std::move(context);
        break;
    }
    default:
        break;
    }
    delete[] trtModelStream;
}


void Rotation::initDetection(){
    yolo_ = std::move(yolo::load(detection_path_, yolo::Type::V8Seg, CONF_THRESHOLD_));
    if (yolo_ == nullptr) return;
}


yolo::Image Rotation::cvimg(const cv::Mat &image) {
    return yolo::Image(image.data, image.cols, image.rows);
}


std::vector<std::string> Rotation::listFiles(const std::string& directory,const std::string & ext) {

    std::vector<std::string> total_names;

    std::experimental::filesystem::path p(directory);
    for(auto & entry : std::experimental::filesystem::directory_iterator(p)){
        if(entry.path().extension().string() == ext){
            total_names.push_back(entry.path().string());
        }
    }
    return total_names;
}


cv::Mat Rotation::preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(127, 127, 127));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    padw = (input_w - w) / 2;
    padh = (input_h - h) / 2;
    return out;
}


void  Rotation::letterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
                          bool autoShape , bool scaleFill  , bool scaleUp , int stride , const cv::Scalar& color )
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}



void Rotation::runGround() {

  auto total_names = listFiles(dji_img_path_, ".JPG");

  for(auto img_path : total_names){

        cv::Mat image = cv::imread(img_path);
        int imageWidth = image.cols;
        int imageHeight = image.rows;
        int k = 900;
        int stride = 700;

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int>indices;
        std::vector<int>label_indexs;

        int newcnt = 0;
        for (int x = 0; x < imageWidth; x += stride) {
            for (int y = 0; y < imageHeight; y += stride) {

                int currentK = std::min(k, imageWidth - x);
                int currentStride = std::min(k, imageHeight - y);

                cv::Rect roi(x, y, currentK, currentStride);
                cv::Mat croppedImage(image, roi);

                std::string out_path = cropped_img_path_+ "/"+std::to_string(newcnt)+".jpg";
                cv::imwrite(out_path,croppedImage);
                newcnt++;

                cv::Mat small_img = cv::imread(out_path);

                auto objs = yolo_->forward(cvimg(small_img));

                for(auto & obj : objs){
                    if(obj.class_label == 4) continue;//rail TODO
                        obj.left += x;
                        obj.top +=y;
                        obj.right+=x;
                        obj.bottom +=y;

                        cv::Rect2i org_rect = cv::Rect(cv::Point2i(obj.left,obj.top),cv::Point2i(obj.right,obj.bottom));
                        float score = obj.confidence;
                        int label = obj.class_label;

                        bboxes.push_back(org_rect);
                        scores.push_back(score);
                        label_indexs.push_back(label);
                }
            }
        }

        cv::dnn::NMSBoxes(bboxes,scores,CONF_THRESHOLD_,NMS_THRESHOLD_,indices);

        int last_slash_pos = img_path.find_last_of("/\\");
        int last_dot_pos = img_path.find_last_of(".");
        std::string img_name = img_path.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);

        json j;
        j["version"] = "0.3.3";
        j["flags"]={};
        j["shapes"]={};
        j["imagePath"]= img_name + ".JPG";
        j["imageData"] ={};
        j["imageHeight"] =imageHeight ;
        j["imageWidth"] = imageWidth;

        for(auto idx: indices){
            int label_index= label_indexs[idx];
            std::string label = class_names_det_[label_index];
            cv::Rect rec = bboxes[idx];
            auto a = rec.tl();
            auto b = rec.br();

            json j_temp;
            j_temp["label"] = label;
            j_temp["text"]="";
            j_temp["points"] ={{a.x,a.y},{b.x,b.y}};
            j_temp["group_id"]={};
            j_temp["shape_type"]="rectangle";
            j_temp["flags"] = {};
            j["shapes"].push_back(j_temp);
        }

        rotateInference(image, j);

        if(!j["shapes"].empty()){
            std::string out_json = dji_img_path_ +"/" + img_name +  + ".json";
            std::ofstream o(out_json);
            o << std::setw(4) << j << std::endl;
        }
    }

}

void Rotation::rotateInference(cv::Mat & img, json &j){

    int32_t input_index = engine_obb_->getBindingIndex(images_);
    int32_t output0_index = engine_obb_->getBindingIndex(output0_);

    void* buffers[2];
    cudaMalloc(&buffers[input_index], OBB_BATCH_SIZE_ * OBB_CHANNELS_ * OBB_INPUT_W_ * OBB_INPUT_H_ * sizeof(float));
    cudaMalloc(&buffers[output0_index], OBB_BATCH_SIZE_ * OBB_OUTPUT0_BOXES_ * OBB_OUTPUT0_ELEMENT_ * sizeof(float));

    cv::Mat pr_img;
    cv::Vec4d params;
    letterBox(img, pr_img, params, cv::Size(OBB_INPUT_W_, OBB_INPUT_H_),false,false,true,32,cv::Scalar(127, 127, 127));

    float *in_arr = new float[1 * 3 * 1024 * 1024];
    float *pdata = new float[1 * 21504 * 7];

    for (int i = 0; i < OBB_INPUT_W_ * OBB_INPUT_H_; i++) {
        in_arr[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        in_arr[i + OBB_INPUT_W_ * OBB_INPUT_H_] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        in_arr[i + 2 * OBB_INPUT_W_ * OBB_INPUT_H_] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], in_arr, OBB_BATCH_SIZE_ * OBB_CHANNELS_ * OBB_INPUT_W_ * OBB_INPUT_H_ * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_obb_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(pdata, buffers[output0_index], OBB_BATCH_SIZE_ * OBB_OUTPUT0_BOXES_ * OBB_OUTPUT0_ELEMENT_ * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output0_index]);

    float r_w = OBB_INPUT_W_ / (img.cols * 1.0);
    float r_h = OBB_INPUT_H_ / (img.rows * 1.0);

    int net_width = OBB_OUTPUT0_ELEMENT_;

    std::vector<cv::RotatedRect> bboxes;
    std::vector<float> scores;
    std::vector<int>indices;
    std::vector<int>label_idxs;

    for (int i = 0; i < OBB_OUTPUT0_BOXES_; i++) {

        float* score_ptr = std::max_element(pdata + 4, pdata + 4 + OBB_CLASSES_);
        float box_score = *score_ptr;
        int label_index = score_ptr - (pdata + 4);

        if (box_score >= CONF_THRESHOLD_) {

            float x_org = (pdata[0] - params[2]) / params[0] ;
            float y_org = (pdata[1] - params[3]) / params[1] ;
            float w = pdata[2] / params[0];
            float h = pdata[3] / params[1];
            float angle = pdata[4+OBB_CLASSES_] /CV_PI *180.0;

            cv::RotatedRect rotate_rect = cv::RotatedRect(cv::Point2f(x_org,y_org),cv::Size2f(w,h), angle);
            bboxes.push_back(rotate_rect);
            scores.push_back(box_score);
            label_idxs.push_back(label_index);

        }

        pdata += net_width; 
    }

    cv::dnn::NMSBoxes(bboxes,scores,CONF_THRESHOLD_,NMS_THRESHOLD_,indices);
                
    for(auto idx : indices){

        int label_index= label_idxs[idx];
        std::string label = class_names_rot_[label_index];
        cv::RotatedRect rec = bboxes[idx];
        float score = scores[idx];

        cv::Point2f ps[4] ={};
        rec.points(ps);

        json j_temp;
        j_temp["label"] = label;
        j_temp["text"]="";
        j_temp["group_id"]={};
        j_temp["shape_type"]="polygon";
        j_temp["flags"] = {};
        j_temp["points"] ={}; 
        for(auto & p : ps){
            j_temp["points"].push_back({p.x, p.y});
        }
        
        j["shapes"].push_back(j_temp);

    }

    delete[] in_arr;
    delete[] pdata;

}


std::vector<CsvInfo> Rotation::readCsv(std::string & csv_path){

    std::fstream file_stream(csv_path);
    std::string str;
    std::vector<CsvInfo> res;
    while (std::getline(file_stream, str)) {

        std::regex reg(",");
        std::sregex_token_iterator beg(str.begin(), str.end(), reg, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> splits(beg, end);

        CsvInfo csv_info;
        csv_info.img_path  = splits[0];
        csv_info.longitude = std::stod(splits[1]);
        csv_info.latitude  = std::stod(splits[2]);
        csv_info.altitude  = std::stod(splits[3]);
        res.push_back(csv_info);

    }
    
    return res;

}



void Rotation::initParams(HyperParams & params){

    csv_path_ = params.csv_path;
    second_csv_path_ = params.second_csv_path;
    json_path_ = params.json_path;
    first_img_path_ = params.first_img_path;
    second_img_path_ = params.second_img_path;

    csvinfo_results_ = readCsv(csv_path_);
    second_csvinfo_results_ = readCsv(second_csv_path_);
    //resize_ratio_ = params.resize_ratio;
}



std::vector<SingleResult> Rotation::runSky(){
    //第二次巡检得到的图片名、经度、维度。相机SKD使用TCP传过来，现用模拟的数据替代。
    // CsvInfo second_info;

    std::vector<SingleResult> total_results;
    for(CsvInfo & second_info : second_csvinfo_results_){

        CsvInfo first_info = findFirstInfo(second_info);

        cv::Mat img_object, img_scene;
        cv::Mat obj_temp, scene_temp;

        std::string img1_path = first_img_path_ + "/" + first_info.img_path;
        std::string img2_path = second_img_path_+ "/" + second_info.img_path;
        std::thread t1( &Rotation::readAndPreprocess, this, std::ref(img_object), std::ref(obj_temp),   std::ref(img1_path));
        std::thread t2( &Rotation::readAndPreprocess, this, std::ref(img_scene),  std::ref(scene_temp), std::ref(img2_path));
        t1.join();
        t2.join();
        SingleResult single_result;
        cv::Mat H = cudaORB(obj_temp, scene_temp);
        if(H.empty()){
            single_result.is_valid = false;
            total_results.push_back(single_result);
            continue;
        }

        std::vector<Location> locations = firstToSecond(H, first_info);
        skyInference(locations, img_scene);

        single_result.img_path = second_info.img_path;
        single_result.longitude = second_info.longitude;
        single_result.latitude = second_info.latitude;
        single_result.altitude = second_info.altitude;
        single_result.locations = locations;
        single_result.is_valid = true;
        total_results.push_back(single_result);
        drawSingleResult(single_result, img_scene);

    }
    return total_results;

}



CsvInfo Rotation::findFirstInfo(CsvInfo & second_info){

    double min_distance = INT32_MAX;
    int min_index = 0;
    for(int i = 0; i < csvinfo_results_.size(); i++) {
        
        double relative_distance = std::sqrt( std::pow(second_info.latitude - csvinfo_results_[i].latitude,2) + std::pow(second_info.longitude - csvinfo_results_[i].longitude,2));
        if(relative_distance < min_distance){
            min_distance = relative_distance;
            min_index = i;
        }
    }

    CsvInfo first_info = csvinfo_results_[min_index];

    return first_info;

}


void Rotation::readAndPreprocess(cv::Mat & img, cv::Mat & temp, std::string & img_path) {

        img = cv::imread(img_path);
        cv::resize(img, temp, cv::Size(), resize_ratio_, resize_ratio_);
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);

}



cv::Mat Rotation::ORB(cv::Mat & img_object, cv::Mat & img_scene){

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(nfeatures_);
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

    cv::Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, cv::noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, cv::noArray(), keypoints_scene, descriptors_scene );

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);


    std::vector< std::vector<cv::DMatch> > knn_matches;

    matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
        good_matches.push_back(knn_matches[i][0]);
        }
    }

    if(good_matches.empty()){
        cv::Mat empty_mat;
        return empty_mat;
    }

    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ ){

    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    cv::Mat H = findHomography( obj, scene, cv::RANSAC );

    return H;

}


cv::Mat Rotation::cudaORB(cv::Mat & img_object, cv::Mat & img_scene){

	cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create(nfeatures_);
	
	cv::cuda::GpuMat gpu_img_object(img_object);
	cv::cuda::GpuMat gpu_img_secne(img_scene);

	cv::cuda::GpuMat gpu_keypoints_object;
	cv::cuda::GpuMat gpu_keypoints_scene;
	cv::cuda::GpuMat descriptors_object, descriptors_scene;
	detector->detectAndComputeAsync( gpu_img_object, cv::cuda::GpuMat(), gpu_keypoints_object, descriptors_object );
	detector->detectAndComputeAsync( gpu_img_secne, cv::cuda::GpuMat(), gpu_keypoints_scene, descriptors_scene );

	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
	detector->convert(gpu_keypoints_object, keypoints_object);
	detector->convert(gpu_keypoints_scene, keypoints_scene);

	cv::cuda::GpuMat gpu_matches;
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
	matcher->knnMatchAsync(descriptors_object, descriptors_scene, gpu_matches, 2);

	std::vector<std::vector<cv::DMatch>> knn_matches;
	matcher->knnMatchConvert(gpu_matches, knn_matches);

	const float ratio_thresh = 0.75f;
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
		good_matches.push_back(knn_matches[i][0]);
		}
	}

    if(good_matches.empty()){
        cv::Mat empty_mat;
        return empty_mat;
    }

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	for( size_t i = 0; i < good_matches.size(); i++ ){
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	cv::Mat H = findHomography( obj, scene, cv::RANSAC );

    return H;
    
}



std::vector<Location> Rotation::firstToSecond(cv::Mat & H, CsvInfo & first_info) {

    std::string name =  first_info.img_path.substr(0, first_info.img_path.find_first_of("."));
    std::string first_json_path = json_path_ + "/" + name + ".json";

//  std::string first_json_path = "/home/nvidia/wjd/sift/jsonpath/DJI_20240227105818_0037.json";

    std::ifstream json_file(first_json_path); 
    json j = json::parse(json_file);

    int height = j["imageHeight"];
    int width = j["imageWidth"];

    std::vector<Location> second_locations;

    for(auto &item : j["shapes"]){

        std::vector<cv::Point2f> pts;
        std::vector<cv::Point2f> transform_pts;
        Location location;
        location.label = item["label"];

        if(item["shape_type"]=="rectangle"){

            auto left  = (float)item["points"][0][0] * resize_ratio_;
            auto top   = (float)item["points"][0][1] * resize_ratio_;
            auto right = (float)item["points"][1][0] * resize_ratio_;
            auto bottom =(float)item["points"][1][1] * resize_ratio_;

            auto width = (float)right -(float) left;
            auto height =(float) bottom - (float)right;
            auto p2_x = right;
            auto p2_y =top;
            auto p4_x = left;
            auto p4_y = bottom;

            pts.emplace_back(cv::Point2f(left, top));
            pts.emplace_back(cv::Point2f(p2_x, p2_y));
            pts.emplace_back(cv::Point2f(right, bottom));
            pts.emplace_back(cv::Point2f(p4_x, p4_y));

        }else if(item["shape_type" ]== "polygon"){

            for(auto & p : item["points"]){
                pts.emplace_back(cv::Point2f((float)p[0] * resize_ratio_, (float)p[1] * resize_ratio_));
            }
            
        }

        cv::perspectiveTransform(pts, transform_pts, H);

        bool valid = true;
        for(auto & p: transform_pts){
            p.x /= resize_ratio_;
            p.y /= resize_ratio_;

            if(location.label != "rail_big"){
                if(p.x<0 || p.x >=width || p.y <0 || p.y>=height){
                    valid = false;
                    break;
                }
            }
        
            location.pts.emplace_back(p);

        }
        if(!valid) continue;

        second_locations.emplace_back(location);

    }

    return second_locations;

}


void Rotation::skyInference( std::vector<Location> & locations, cv::Mat & img){

    for(auto & i : locations){

        if(i.label != "rail_big"){

            cv::RotatedRect rotated_rect =  cv::minAreaRect (i.pts) ;

            cv::Point2f *pts = new cv::Point2f[i.pts.size()];

            rotated_rect.points(pts);

            int width = rotated_rect.size.width;
            int height = rotated_rect.size.height;

            cv::Point2f * dst_pts;

            dst_pts = new cv::Point2f[4]{ cv::Point2f(0, height), cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(width, height) };
            cv::Mat M = cv::getPerspectiveTransform(pts, dst_pts);
            cv::Mat dst_img;

            cv::warpPerspective(img, dst_img, M, cv::Size(width, height));

            if(width<height){
                 cv::rotate(dst_img, dst_img, cv::ROTATE_90_CLOCKWISE);
            }


            int type = doClassisication(dst_img, i.label);
            i.type = type;

            delete []pts;
            delete []dst_pts;

        }

    }

}


int Rotation::doClassisication(cv::Mat & img, std::string & label){
    int res = 1;

    if(label == "sleeper_normal" || label == "sleeper_abnormal"){
        res = sleeperClassification(img);
    }else if(label == "fastener_normal" || label == "fastener_abnormal"){
        res = fastenerClassification(img);
    }else{
        return 1;
    }
    return res;
}


//int sleeper_index = 0;
int Rotation::sleeperClassification(cv::Mat & img){

    cv::Mat img_pad;
    cv::Vec4d params;    
    letterBox(img, img_pad, params, cv::Size(SLEEPER_INPUT_W_, SLEEPER_INPUT_H_));
//    cv::imwrite("F:/PLATFORM/tools/sift/testcsv/testimg/"+std::to_string(sleeper_index)+"sleeper.jpg",img_pad);
//    sleeper_index++;

    int input_index = engine_sleeper_->getBindingIndex(INPUT_NAMES_);
    int output_index = engine_sleeper_->getBindingIndex(OUTPUT_NAMES_);

    void* buffers[2];
    cudaMalloc(&buffers[input_index], BATCH_SIZE_ * CHANNELS_ * SLEEPER_INPUT_W_ * SLEEPER_INPUT_H_ * sizeof(float));
    cudaMalloc(&buffers[output_index], BATCH_SIZE_ * CLASSES_ * sizeof(float));

    for (int i = 0; i < SLEEPER_INPUT_W_ * SLEEPER_INPUT_H_ ; i++) {
        sleeper_cls_in_[i] = img_pad.at<cv::Vec3b>(i)[2] / 1.0;
        sleeper_cls_in_[i + SLEEPER_INPUT_W_ * SLEEPER_INPUT_H_ ] = img_pad.at<cv::Vec3b>(i)[1] / 1.0;
        sleeper_cls_in_[i + 2 * SLEEPER_INPUT_W_ * SLEEPER_INPUT_H_ ] = img_pad.at<cv::Vec3b>(i)[0] / 1.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], sleeper_cls_in_, BATCH_SIZE_ * CHANNELS_ * SLEEPER_INPUT_W_ * SLEEPER_INPUT_H_  * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_sleeper_->enqueueV2(buffers, stream, nullptr);

    cudaMemcpyAsync(sleeper_cls_out_, buffers[output_index], BATCH_SIZE_ * CLASSES_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);

    int index = std::max_element( sleeper_cls_out_, sleeper_cls_out_ + CLASSES_ ) - sleeper_cls_out_;

    //std::cout<<"abnormal sleeper ="<<sleeper_cls_out_[0]<<" normal sleeper = "<<sleeper_cls_out_[1]<<std::endl;
    
    return index;
}

//int fastener_index = 0;
int Rotation::fastenerClassification(cv::Mat & img){

    cv::Mat img_pad;
    cv::Vec4d params;    
    letterBox(img, img_pad, params, cv::Size(FASTENER_INPUT_W_, FASTENER_INPUT_H_));

//    cv::imwrite("F:/PLATFORM/tools/sift/testcsv/testimg/"+std::to_string(fastener_index)+"fastener.jpg",img_pad);
//    fastener_index++;

    int input_index = engine_fastener_->getBindingIndex(INPUT_NAMES_);
    int output_index = engine_fastener_->getBindingIndex(OUTPUT_NAMES_);

    void* buffers[2];
    cudaMalloc(&buffers[input_index], BATCH_SIZE_ * CHANNELS_ * FASTENER_INPUT_W_ * FASTENER_INPUT_H_ * sizeof(float));
    cudaMalloc(&buffers[output_index], BATCH_SIZE_ * CLASSES_ * sizeof(float));

    for (int i = 0; i < FASTENER_INPUT_W_ * FASTENER_INPUT_H_ ; i++) {
        fastener_cls_in_[i] = img_pad.at<cv::Vec3b>(i)[2] / 1.0;
        fastener_cls_in_[i + FASTENER_INPUT_W_ * FASTENER_INPUT_H_ ] = img_pad.at<cv::Vec3b>(i)[1] / 1.0;
        fastener_cls_in_[i + 2 * FASTENER_INPUT_W_ * FASTENER_INPUT_H_ ] = img_pad.at<cv::Vec3b>(i)[0] / 1.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], fastener_cls_in_, BATCH_SIZE_ * CHANNELS_ * FASTENER_INPUT_W_ * FASTENER_INPUT_H_  * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_fastener_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(fastener_cls_out_, buffers[output_index], BATCH_SIZE_ * CLASSES_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);

    int index = std::max_element( fastener_cls_out_, fastener_cls_out_ + CLASSES_ ) - fastener_cls_out_;
    //std::cout<<"abnormal fastener ="<<fastener_cls_out_[0]<<" normal fastener = "<<fastener_cls_out_[1]<<std::endl;

    return index;
}



void Rotation::testSky(){

        cv::Mat img_object, img_scene;
        cv::Mat obj_temp ,scene_temp;

        std::string path1 = "/home/nvidia/wjd/sift/jsonpath/DJI_20240227105818_0037.JPG";
        std::string path2 = "/home/nvidia/wjd/sift/jsonpath/DJI_20240227105819_0038.JPG";
        std::thread t1( &Rotation::readAndPreprocess,this,std::ref(img_object), std::ref(obj_temp)  , std::ref(path1));
        std::thread t2( &Rotation::readAndPreprocess, this,std::ref(img_scene) , std::ref(scene_temp), std::ref(path2));
        t1.join();
        t2.join();

        CsvInfo first_info;
        cv::Mat H = cudaORB(obj_temp, scene_temp);

        std::vector<Location> locations = firstToSecond(H, first_info);

        //drawSecond(locations, img_scene);

        skyInference(locations, img_scene);

}

void Rotation::drawSecond(std::vector<Location> & locations, cv::Mat & img){

    cv::Mat total_mask = cv::Mat::zeros(HEIGHT_, WIDTH_, CV_8UC1);
    for(auto i : locations) {
        std::string label = i.label;
        int len = i.pts.size();
        for(int j = 0 ; j < len; j++){
            auto p1 = i.pts[j % len ];
            auto p2 = i.pts[(j+1) % len];
            cv::line(img ,p1, p2, cv::Scalar(255,0,0),2);
        }
        cv::putText(img, label, i.pts[0],2 ,2 ,cv::Scalar(0,0,255));

        if(label == "rail_big"){
            std::vector<cv::Point> points;
            for(auto & p : i.pts){
                points.push_back(cv::Point(p));
            }
            cv::fillPoly(total_mask, points, cv::Scalar(255));

        }

    }
    cv::imwrite("../test.jpg",img);

    cv::Mat result_img = cv::Mat::zeros(HEIGHT_, WIDTH_, CV_8UC3);
    cv::bitwise_and(img, img, result_img, total_mask );

    cv::imwrite("../rail.jpg",result_img);
    cv::imwrite("../total_mask.jpg",total_mask);

}


void Rotation::testORB(){

    auto read_time_start = std::chrono::system_clock::now();
    cv::Mat img_object = cv::imread("../dajiang1.bmp");
    cv::Mat img_scene = cv::imread("../dajiang3.bmp");
    auto read_time_end = std::chrono::system_clock::now();
    auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(read_time_end - read_time_start).count();
    std::cout<<"read_time = "<<read_time<<std::endl;

    auto resize_time_start = std::chrono::system_clock::now();
    cv::resize(img_object,img_object,cv::Size(),resize_ratio_, resize_ratio_);
    cv::resize(img_scene, img_scene,cv::Size(),resize_ratio_, resize_ratio_);
    auto resize_time_end = std::chrono::system_clock::now();
    auto resize_time = std::chrono::duration_cast<std::chrono::milliseconds>(resize_time_end - resize_time_start).count();
    std::cout<<"resize_time = "<<resize_time<<std::endl;

    auto convert_time_start = std::chrono::system_clock::now();
    cv::cvtColor(img_object, img_object, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_scene,  img_scene, cv::COLOR_BGR2GRAY);
    auto convert_time_end = std::chrono::system_clock::now();
    auto convert_time = std::chrono::duration_cast<std::chrono::milliseconds>(convert_time_end - convert_time_start).count();
    std::cout<<"convert_time = "<<convert_time<<std::endl;

    auto start = std::chrono::system_clock::now();

    auto H_cuda = cudaORB(img_object, img_scene);
    std::cout<<"H_cuda.empty = "<<H_cuda.empty()<<std::endl;
    std::cout<<"H_cuda = \n"<<H_cuda<<std::endl;

    auto end1 = std::chrono::system_clock::now();

    auto H_cpu = ORB(img_object, img_scene);
    std::cout<<"H_cpu.empty = "<<H_cpu.empty()<<std::endl;
    std::cout<<"H_cpu = \n"<<H_cpu<<std::endl;

    auto end2 = std::chrono::system_clock::now();

    auto pass_time_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end2-end1).count();

    auto pass_time_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(end1-start).count();

    std::cout<<"pass_time_cuda = "<<pass_time_cuda<<std::endl;
    std::cout<<"pass_time_cpu = "<<pass_time_cpu<<std::endl;

}



static int sleeper_cnt = 18000;
static int fastener_cnt = 18000;
void Rotation::readJson(){

    std::set<std::string> temp_set;
    std::experimental::filesystem::path p = "/home/nvidia/wjd/djiimg";
    for(auto & entry : std::experimental::filesystem::directory_iterator(p)){
        if(std::experimental::filesystem::is_regular_file(entry)){

            if(entry.path().extension().string() == ".json"){

                    std::string img_name = entry.path().stem().string() + ".JPG";
                    std::string img_path = entry.path().parent_path().string() + "/" + img_name;

                    cv::Mat img = cv::imread(img_path);

                    std::string json_path = entry.path().string();

                    std::ifstream json_file(json_path);
                    json j = json::parse(json_file);

                    int height = j["imageHeight"];
                    int width = j["imageWidth"];


                    for(auto &item : j["shapes"]){

                        std::string label = item["label"];
                        temp_set.insert(label);
                        if(label == "rail_big") continue;

                        std::string type = item["shape_type"];

                        std::vector<cv::Point2f> before_pts;
                        if(type == "rectangle"){

                            auto left  = (float)item["points"][0][0];
                            auto top   = (float)item["points"][0][1];
                            auto right = (float)item["points"][1][0];
                            auto bottom =(float)item["points"][1][1];

                            auto width = (float)right -(float) left;
                            auto height =(float) bottom - (float)right;
                            auto p2_x = right;
                            auto p2_y =top;
                            auto p4_x = left;
                            auto p4_y = bottom;

                            before_pts.emplace_back(cv::Point2f(left, top));
                            before_pts.emplace_back(cv::Point2f(p2_x, p2_y));
                            before_pts.emplace_back(cv::Point2f(right, bottom));
                            before_pts.emplace_back(cv::Point2f(p4_x, p4_y));

                        }else if(type=="polygon"){

                            for(auto i: item["points"]){

                                before_pts.emplace_back(cv::Point2f((float)i[0], (float)i[1]));

                            }


                        }

                        cv::RotatedRect rotated_rect = cv::minAreaRect(before_pts) ;

                        cv::Point2f *pts = new cv::Point2f[before_pts.size()];

                        rotated_rect.points(pts);

                        int width = rotated_rect.size.width;
                        int height = rotated_rect.size.height;

                        cv::Point2f * dst_pts;

                        dst_pts = new cv::Point2f[4]{ cv::Point2f(0, height), cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(width, height) };
                        cv::Mat M = cv::getPerspectiveTransform(pts, dst_pts);
                        cv::Mat dst_img;

                        cv::warpPerspective(img, dst_img, M, cv::Size(width, height));

                        if(width<height){
                            cv::rotate(dst_img, dst_img, cv::ROTATE_90_CLOCKWISE);
                        }


                        if(label=="fastener_abnormal" || label == "fastener_missing" || label == "fastener_stone" ){
                            cv::imwrite("/home/nvidia/wjd/sss/fastener_abnormal/"+std::to_string(fastener_cnt)+".jpg",dst_img);
                            fastener_cnt++;

                        }
                        if(label == "fastener_normal" ){

                            cv::imwrite("/home/nvidia/wjd/sss/fastener_normal/"+std::to_string(fastener_cnt)+".jpg",dst_img);
                            fastener_cnt++;
                        }

                        if(label == "sleeper_normal" ){

                            cv::imwrite("/home/nvidia/wjd/sss/sleeper_normal/"+std::to_string(sleeper_cnt)+".jpg",dst_img);
                            sleeper_cnt++;

                        }

                        if(label == "sleeper_abnormal" ){

                            cv::imwrite("/home/nvidia/wjd/sss/sleeper_abnormal/"+std::to_string(sleeper_cnt)+".jpg",dst_img);
                            sleeper_cnt++;

                        }

                        delete[] pts;
                        delete[] dst_pts;

                    }
            }
        }
    }
    for(auto i: temp_set){
        std::cout<<i<<std::endl;
    }
}



void Rotation::drawSingleResult(SingleResult & single_result, cv::Mat& img){

  std::string save_prefix = save_prefix_;
  std::string img_path = save_prefix + "/" + single_result.img_path;
  if(!single_result.is_valid){

      cv::imwrite(img_path, img);

  }else{

      for(auto & j: single_result.locations){

          std::string label = j.label;
          int type = j.type;
          for(int k = 0; k < j.pts.size(); k++){
              if(!type){
                  cv::line(img, j.pts[k], j.pts[ (k+1)%j.pts.size()],cv::Scalar(0,0,255),3,1);
              }else{

                  cv::line(img, j.pts[k], j.pts[ (k+1)%j.pts.size()],cv::Scalar(255,0,0),3,1);
              }
          }
          if(!j.pts.empty()){

             cv::putText(img, label, j.pts[0],2,1,cv::Scalar(0,0,0),2);

          }

      }
      cv::imwrite(img_path,img);

  }

}


void Rotation::testRunSky(){

    HyperParams params;
    params.csv_path = "F:/PLATFORM/tools/sift/testcsv/testcsv_template/template.csv";
    params.second_csv_path = "F:/PLATFORM/tools/sift/testcsv/testcsv_inspection/inspection.csv";
    params.json_path = "F:/PLATFORM/tools/sift/testcsv/testcsv_template";
    params.first_img_path = "F:/PLATFORM/tools/sift/testcsv/testcsv_template";
    params.second_img_path = "F:/PLATFORM/tools/sift/testcsv/testcsv_inspection";

    std::string fastener_path = "F:/PLATFORM/tools/sift/models/fastener.engine";
    std::string sleeper_path = "F:/PLATFORM/tools/sift/models/sleeper.engine";

    this->initParams(params);
    this->loadEngine(sleeper_path, 1);
    this->loadEngine(fastener_path, 2);
    auto res = this->runSky();

}

void Rotation::initGroundSky(){

        ini::iniReader config;
        bool ret = config.ReadConfig("../sift/config/config.ini");
        if(!ret){
            printf("initial failed!\n");
            return;
        };
        //sky
        std::string first_csv_path = config.ReadString("sky", "first_csv_path", "");
        std::string second_csv_path = config.ReadString("sky","second_csv_path", "");
        std::string first_json_path = config.ReadString("sky", "first_json_path", "");
        std::string first_img_path = config.ReadString("sky", "first_img_path", "");
        std::string second_img_path = config.ReadString("sky", "second_img_path", "");
        std::string save_second_img_path = config.ReadString("sky","save_second_img_path","");
        std::string fastener_path = config.ReadString("sky","fastener_path", "");
        std::string sleeper_path = config.ReadString("sky", "sleeper_path", "");
        csv_path_ = first_csv_path;
        second_csv_path_ = second_csv_path;
        json_path_ = first_json_path;
        first_img_path_ = first_img_path;
        second_img_path_ = second_img_path;
        save_prefix_ = save_second_img_path;
        csvinfo_results_ = readCsv(csv_path_);
        second_csvinfo_results_ = readCsv(second_csv_path_);
        loadEngine(sleeper_path ,1);
        loadEngine(fastener_path, 2);

        //ground
        std::string dji_img_path = config.ReadString("ground", "dji_img_path","");
        std::string sleeper_detection_path = config.ReadString("ground", "sleeper_detection_path","");
        std::string fastener_detection_path = config.ReadString("ground", "fastener_detection_path", "");
        std::string cropped_img_path = config.ReadString("ground", "cropped_img_path","");
        float confidence_threshold = config.ReadFloat("ground", "confidenct_threshold", 0.4);
        float nms_threshold = config.ReadFloat("ground", "nms_threshold", 0.25);
        detection_path_ = fastener_detection_path;
        obb_path_ = sleeper_detection_path;
        dji_img_path_ = dji_img_path;
        cropped_img_path_ = cropped_img_path;
        CONF_THRESHOLD_ = confidence_threshold;
        NMS_THRESHOLD_ = nms_threshold;
        //loadEngine(sleeper_detection_path, 0);
        //initDetection();

}



