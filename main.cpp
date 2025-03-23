#include "AnythingCppInference.h"

#define TEST_ROOT "D:/project/personal/anything-cpp-inference/models"

int test_Det_ONNX_MMYOLO()
{
    std::cout << "*****************" << "\n";
    std::cout << __FUNCTION__ << ":\n";
    BaseInference *handle = nullptr;
    char model_dir[100] = TEST_ROOT;
    strcat(model_dir, "/mmyolo_yolov8n");
    bool init_success = ACI::InitModel(handle, ACI::ModelType::Det_ONNX_MMYOLO, model_dir);
    if (init_success)
        std::cout << "Init Success." << "\n";
    else
        std::cout << "Init Failed" << "\n";
    AiData::InnerModelInput inp;
    const char *img_path = (std::string(TEST_ROOT) + "/demo.jpg").c_str();
    inp.img = cv::imread(img_path, cv::IMREAD_COLOR);
    AiData::InnerModelOutput oup;
    bool inference_success = ACI::Inference(handle, inp, oup);
    if (inference_success)
        std::cout << "Inference Success." << "\n";
    else
        std::cout << "Inference Failed" << "\n";
    cv::Mat img_vis = inp.img.clone();
    for (int i = 0; i < oup.det_result.n_boxes; ++i)
    {
        float x1 = *(oup.det_result.boxes + i * 4);
        float y1 = *(oup.det_result.boxes + i * 4 + 1);
        float x2 = *(oup.det_result.boxes + i * 4 + 2);
        float y2 = *(oup.det_result.boxes + i * 4 + 3);
        cv::rectangle(img_vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
    }
    const char *save_path = (std::string(TEST_ROOT) + "/" + __FUNCTION__ + ".jpg").c_str();
    cv::imwrite(save_path, img_vis);
    std::cout << "*****************" << "\n";
    return 0;
}

int test_Det_ONNX_UltralyticsYolo()
{
    std::cout << "*****************" << "\n";
    std::cout << __FUNCTION__ << ":\n";
    BaseInference *handle = nullptr;
    char model_dir[100] = TEST_ROOT;
    strcat(model_dir, "/ultralytics_yolov8s");
    bool init_success = ACI::InitModel(handle, ACI::ModelType::Det_ONNX_UltralyticsYolo, model_dir);
    if (init_success)
        std::cout << "Init Success." << "\n";
    else
        std::cout << "Init Failed" << "\n";
    AiData::InnerModelInput inp;
    const char *img_path = (std::string(TEST_ROOT) + "/demo.jpg").c_str();
    inp.img = cv::imread(img_path, cv::IMREAD_COLOR);
    AiData::InnerModelOutput oup;
    bool inference_success = ACI::Inference(handle, inp, oup);
    if (inference_success)
        std::cout << "Inference Success." << "\n";
    else
        std::cout << "Inference Failed" << "\n";
    cv::Mat img_vis = inp.img.clone();
    for (int i = 0; i < oup.det_result.n_boxes; ++i)
    {
        float x1 = *(oup.det_result.boxes + i * 4);
        float y1 = *(oup.det_result.boxes + i * 4 + 1);
        float x2 = *(oup.det_result.boxes + i * 4 + 2);
        float y2 = *(oup.det_result.boxes + i * 4 + 3);
        cv::rectangle(img_vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
    }
    const char *save_path = (std::string(TEST_ROOT) + "/" + __FUNCTION__ + ".jpg").c_str();
    cv::imwrite(save_path, img_vis);
    std::cout << "*****************" << "\n";
    return 0;
}

int main()
{
    test_Det_ONNX_MMYOLO();
    test_Det_ONNX_UltralyticsYolo();
    return 0;
}