#ifndef BaseInference_H
#define BaseInference_H
#include "pch.h"
#include "base_data.h"
#include "utils.h"

class BaseInference
{
public:
    BaseInference() {};
    ~BaseInference() {};
    bool load_model(const std::string modelDir)
    {
        update_from_config(modelDir);
        update_model();
        return true;
    };
    virtual bool update_from_config(const std::string modelDir);
    virtual bool update_model() = 0;

    virtual bool inference(const AiData::InnerModelInput &input, AiData::InnerModelOutput &output) = 0;
    virtual bool inference(const std::vector<AiData::InnerModelInput> &inputs, std::vector<AiData::InnerModelOutput> &outputs) = 0;

protected:
    virtual bool resize(const cv::Mat &src, cv::Mat &dst, TransformParam &transformParam);
    virtual bool resizeInv(const TransformParam &transformParam, AiData::InnerModelOutput &output);

protected:
    bool m_swapRb = true;
    int m_downSampleBase = 32;
    bool m_keepRatio = false;
    bool m_PadAfterResize = false;

    float m_mean[3] = {0.0f, 0.0f, 0.0f};
    float m_std[3] = {1.0f / 255, 1.0f / 255, 1.0f / 255};

protected:
    std::string m_modelPath;
    int m_inputHeight;
    int m_inputWidth;
    int m_channel;
    int m_batchsize;

    myutils::JsonData json_data;
};
#endif