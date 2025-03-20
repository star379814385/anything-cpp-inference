#include "AnythingCppInference.h"
#ifdef USE_ONNX_H
#include "onnx/MMYoloONNXInference.h"
#endif

bool ACI::InitModel(BaseInference *&handle, int model_type, char *config_dir)
{
#ifdef USE_ONNX_H
    if (model_type == ModelType::MMYOLO)
    {
        handle = new MMYoloONNXInference();
    }
#endif
    if (handle == nullptr)
        return false;
    return handle->load_model(config_dir);
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
