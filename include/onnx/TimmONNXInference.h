#include "ONNXInference.h"

class TimmONNXInference : public ONNXInference
{
public:
    TimmONNXInference()
    {
        m_swapRb = true;
        m_downSampleBase = 32;
        m_keepRatio = false;
        m_PadAfterResize = false;
        m_mean[0] = 0.485f * 255;
        m_mean[1] = 0.456f * 255;
        m_mean[2] = 0.406f * 255;
        m_std[0] = 0.229f * 255;
        m_std[1] = 0.224f * 255;
        m_std[2] = 0.225f * 255;
    };
    ~TimmONNXInference() {

    };

protected:
    virtual bool update_from_config(const std::string modelDir)
    {
        ONNXInference::update_from_config(modelDir);
        return true;
    };

protected:
    virtual bool decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs) override;
};