#include "AnythingCppInference.h"

int main()
{
    std::cout << "hello world" << std::endl;
    BaseInference* handle = nullptr;
    char* model_dir = "D:/project/personal/anything-cpp-inference/models/mmyolo_yolov8n";
    bool init_success = ACI::InitModel(handle, ACI::ModelType::MMYOLO, model_dir);
    std::cout << init_success << std::endl;
    AiData::InnerModelInput inp;
    inp.img = cv::imread("D:/project/personal/anything-cpp-inference/data/demo.jpg", cv::IMREAD_COLOR);
    AiData::InnerModelOutput oup;
    ACI::Inference(handle, inp, oup);
    // model.inference(inp, oup);
    std::cout << "hello world" << std::endl;
    // std::cout << oup.det_result.n_boxes << std::endl;
    cv::Mat img_vis = inp.img.clone();
    for(int i = 0; i < oup.det_result.n_boxes; ++i)
    {
        float x1 = *(oup.det_result.boxes + i * 4);
        float y1 = *(oup.det_result.boxes + i * 4 + 1);
        float x2 = *(oup.det_result.boxes + i * 4 + 2);
        float y2 = *(oup.det_result.boxes + i * 4 + 3);
        cv::rectangle(img_vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
    }
    std::cout << oup.det_result.n_boxes << std::endl;
    cv::imwrite("demo_vis.png", img_vis);
    return 0;
}