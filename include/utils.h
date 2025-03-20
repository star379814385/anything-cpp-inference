#include "json/json.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

namespace myutils
{

    class JsonData
    {
    public:
        JsonData() {};
        JsonData(const std::string json_path)
        {

            // 1.打开文件
            std::ifstream f;
            f.open(json_path);

            // 2.创建json读工厂对象
            Json::CharReaderBuilder ReaderBuilder;
            ReaderBuilder["emitUTF8"] = true; // utf8支持，不加这句，utf8的中文字符会编程\uxxx

            // 4.把文件转变为json对象，要用到上面的三个变量,数据写入root
            std::string strerr;
            bool ok = Json::parseFromStream(ReaderBuilder, f, &root, &strerr);
            if (!ok)
            {
                std::cerr << "json解析错误";
            }
        }
        ~JsonData() {};

    public:
        Json::Value root;
    };

    class Timer
    {
    public:
        Timer()
        {
            start_time = std::chrono::steady_clock::now();
        }
        Timer(std::string taskname)
        {
            using namespace std::literals::chrono_literals;
            m_taskname = taskname;
            start_time = std::chrono::steady_clock::now();
        }
        ~Timer()
        {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Task(" << m_taskname << ") cost " << duration.count() << " ms.\n";
        }

    private:
        std::chrono::steady_clock::time_point start_time;
        std::string m_taskname = "";
        std::string m_unit = "ms";
    };
}
