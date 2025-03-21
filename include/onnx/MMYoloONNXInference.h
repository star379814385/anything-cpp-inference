#include "ONNXInference.h"

class MMYoloONNXInference : public ONNXInference
{
public:
    MMYoloONNXInference()
    {
        m_swapRb = true;
        m_downSampleBase = 32;
        m_keepRatio = true;
        m_PadAfterResize = true;
        m_mean[0] = 0.0f;
        m_mean[1] = 0.0f;
        m_mean[2] = 0.0f;
        m_std[0] = 1.0f * 255;
        m_std[1] = 1.0f * 255;
        m_std[2] = 1.0f * 255;
    };
    ~MMYoloONNXInference() {

    };

protected:
    virtual bool update_from_config(const std::string modelDir)
    {
        ONNXInference::update_from_config(modelDir);
        m_scoreThr = json_data.root["score_thr"].asFloat();
        return true;
    };

protected:
    virtual bool decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs);
    float m_scoreThr;
};