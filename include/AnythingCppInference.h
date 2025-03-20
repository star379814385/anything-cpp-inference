#include "BaseInference.h"

namespace ACI
{
    enum ModelType
    {
        MMYOLO = 0
    };

    bool InitModel(BaseInference *&handle, int model_type, char *config_dir);
    bool Inference(BaseInference *handle, const AiData::InnerModelInput &input, AiData::InnerModelOutput &output);
    bool Inference(BaseInference *handle, const std::vector<AiData::InnerModelInput> &input, std::vector<AiData::InnerModelOutput> &output);

};
