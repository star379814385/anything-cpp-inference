#include "BaseInference.h"

namespace ACI
{
    enum ModelType
    {
        Det_ONNX_MMYOLO = 0,
        Det_ONNX_UltralyticsYolo = 1,
        Cls_ONNX_TIMM = 2
    };

    bool InitModel(BaseInference *&handle, const int model_type, const char *config_dir);
    bool Inference(BaseInference *handle, const AiData::InnerModelInput &input, AiData::InnerModelOutput &output);
    bool Inference(BaseInference *handle, const std::vector<AiData::InnerModelInput> &input, std::vector<AiData::InnerModelOutput> &output);

};
