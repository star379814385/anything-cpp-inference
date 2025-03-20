#include "BaseInference.h"
#include "onnxruntime_cxx_api.h"

/*
example
|-- yolov8_coco80
    |-- config.json
    |-- models.onnx
*/
class ONNXInference : public BaseInference
{
public:
    ONNXInference(){};
    ~ONNXInference(){};
    // virtual bool inference(const AiData::InnerModelInput &input, AiData::InnerModelOutput &output) override;
    virtual bool inference(const AiData::InnerModelInput &input, AiData::InnerModelOutput &output) override;
    virtual bool inference(const std::vector<AiData::InnerModelInput> &inputs, std::vector<AiData::InnerModelOutput> &outputs) override;
protected:
    virtual bool update_from_config(const std::string modelDir)
    {
        BaseInference::update_from_config(modelDir);
        use_cuda = json_data.root["use_cuda"].asBool();
        num_threads = json_data.root["num_threads"].asInt();      
        return true;
    };
    virtual bool update_model() override;
    

protected:
    virtual bool preprocess(const cv::Mat &src, cv::Mat &dst, TransformParam &transformParam);
    virtual bool decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs) = 0;
    // virtual bool postprocess(const std::vector<Ort::Value> &onnx_output, AiData::InnerModelOutput &output, TransformParam transformParam) = 0;
    // virtual bool postprocess(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs, std::vector<TransformParam> transform_param_list) = 0; 
    virtual bool postprocess(AiData::InnerModelOutput &output, const TransformParam &transformParam);

private:
    Ort::Env env;
    Ort::Session *session;
    Ort::RunOptions options;
    std::vector<const char *> inputNodeNames;
    std::vector<const char *> outputNodeNames;
    bool use_cuda;
    int num_threads;
};