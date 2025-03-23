#include "ONNXInference.h"

bool ONNXInference::update_model()
{
    Ort::SessionOptions sessionOption;
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX");
    if (use_cuda)
    {
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(num_threads);
        sessionOption.SetLogSeverityLevel(4);
    }
#ifdef _WIN32
    int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, m_modelPath.c_str(), static_cast<int>(m_modelPath.length()), nullptr, 0);
    wchar_t *wide_cstr = new wchar_t[ModelPathSize + 1];
    MultiByteToWideChar(CP_UTF8, 0, m_modelPath.c_str(), static_cast<int>(m_modelPath.length()), wide_cstr, ModelPathSize);
    wide_cstr[ModelPathSize] = L'\0';
    const wchar_t *modelPath = wide_cstr;
#else
    const char *modelPath = iParams.modelPath.c_str();
#endif // _WIN32
    session = new Ort::Session(env, modelPath, sessionOption);
    delete[] wide_cstr;
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++)
    {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char *temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char *temp_buf = new char[50];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }

    options = Ort::RunOptions{nullptr};
    return true;
}

bool ONNXInference::inference(const AiData::InnerModelInput &input, AiData::InnerModelOutput &output)
{
    std::vector<AiData::InnerModelInput> inputs(1);
    std::vector<AiData::InnerModelOutput> outputs;
    inputs[0].img = input.img;
    bool res = inference(inputs, outputs);
    if (!res)
        return res;
    output = std::move(outputs[0]);
    return true;
}

bool ONNXInference::inference(const std::vector<AiData::InnerModelInput> &inputs, std::vector<AiData::InnerModelOutput> &outputs)
{
    std::vector<cv::Mat> img_process_list(inputs.size());
    std::vector<TransformParam> trans_param_list(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        preprocess(inputs[i].img, img_process_list[i], trans_param_list[i]);
    }
    int img_height = img_process_list[0].rows;
    int img_width = img_process_list[0].cols;
    cv::Mat blob = cv::dnn::blobFromImages(img_process_list, 1.0, {img_width, img_height}, cv::Scalar(0, 0, 0), false, false, CV_32F);
    // to onnx
    std::vector<int64_t> YOLO_input_node_dims = {1, m_channel, img_height, img_width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), (float *)blob.data, inputs.size() * 3 * img_height * img_width, YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
    auto output_tensor = session->Run(options, inputNodeNames.data(), &input_tensor, inputNodeNames.size(), outputNodeNames.data(), outputNodeNames.size());
    decode_result(output_tensor, outputs);
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        postprocess(outputs[i], trans_param_list[i]);
    }
    std::cout << "Inference Done!" << std::endl;
    return true;
}

bool ONNXInference::preprocess(const cv::Mat &src, cv::Mat &dst, TransformParam &transformParam)
{
    resize(src, dst, transformParam);
    if (m_swapRb)
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    dst.convertTo(dst, CV_32FC3);
    std::vector<cv::Mat> img_channels;
    cv::split(dst, img_channels);
    for (size_t i = 0; i < img_channels.size(); ++i)
    {
        img_channels[i] = (img_channels[i] - m_mean[i]) / m_std[i];
    }
    cv::merge(img_channels, dst);
    return true;
}

bool ONNXInference::postprocess(AiData::InnerModelOutput &output, const TransformParam &transformParam)
{
    return resizeInv(transformParam, output);
}
