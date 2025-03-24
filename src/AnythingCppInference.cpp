#include "AnythingCppInference.h"
#ifdef USE_ONNX_H
#include "MMYoloONNXInference.h"
#include "UltralyticsONNXInference.h"
#include "TimmONNXInference.h"
#endif

bool ACI::InitModel(BaseInference *&handle, const int model_type, const char *config_dir)
{
#ifdef USE_ONNX_H
    if (model_type == ModelType::Det_ONNX_MMYOLO)
    {
        handle = new MMYoloONNXInference();
    }
    else if (model_type == ModelType::Det_ONNX_UltralyticsYolo)
    {
        handle = new UltralyticsONNXInference();
    }
    else if (model_type == ModelType::Cls_ONNX_TIMM)
    {
        handle = new TimmONNXInference();
    }

#endif
    if (handle == nullptr)
        return false;
    return handle->load_model(std::string(config_dir));
}

bool ACI::Inference(BaseInference *handle, const AiData::InnerModelInput &input, AiData::InnerModelOutput &output)
{
    myutils::Timer timer("ACI INFERENCE");
    return handle->inference(input, output);
}

bool ACI::Inference(BaseInference *handle, const std::vector<AiData::InnerModelInput> &input, std::vector<AiData::InnerModelOutput> &output)
{
    myutils::Timer timer("ACI INFERENCE");
    return handle->inference(input, output);
}
