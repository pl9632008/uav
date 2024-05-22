#include <iostream>
#include <curl/curl.h>
#include <chrono>


int main(int argc, char* argv[]) {
    // if (argc != 4) {
    //     std::cerr << "Usage: ./ftp_upload <FTP_URL> <USERNAME> <PASSWORD>" << std::endl;
    //     return 1;
    // }


    const char* url = "ftp://192.168.2.32/haha2.JPG";
    const char* username = "ftpperson";
    const char* password = "123456";
    const char* file_to_upload = "/home/ubuntu/wjd/mavtest2/testdji.JPG"; // 要上传的文件路径

    CURL* curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();


    if (curl) {


        // 设置FTP URL
        curl_easy_setopt(curl, CURLOPT_URL, url);

        // 设置FTP上传操作
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

        // 设置用户名和密码
        curl_easy_setopt(curl, CURLOPT_USERPWD, (username + std::string(":") + password).c_str());

        // 设置要上传的文件路径
        curl_easy_setopt(curl, CURLOPT_READDATA, fopen(file_to_upload, "rb"));

     

        auto end2 = std::chrono::system_clock::now();

        res = curl_easy_perform(curl);


        auto end3 = std::chrono::system_clock::now();
        auto passtime3 = std::chrono::duration_cast<std::chrono::seconds>(end3-end2).count();
        std::cout<<"passtime3 = "<<passtime3<<std::endl;

        // 检查执行结果
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        // 清理资源
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();

    return 0;
}
