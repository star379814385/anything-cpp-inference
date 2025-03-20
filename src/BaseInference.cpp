#include "BaseInference.h"


bool BaseInference::update_from_config(const std::string modelDir)
{
    const std::string configPath = modelDir + "/config.json";
    json_data = myutils::JsonData(configPath);
    m_batchsize = 1;
    m_channel = 3;
    m_inputHeight = json_data.root["input_height"].asInt();
    m_inputWidth = json_data.root["input_width"].asInt();
    m_modelPath = modelDir + "/" + json_data.root["model_name"].asString();
    return true;
}

bool BaseInference::resize(const cv::Mat &src, cv::Mat &dst, TransformParam &transformParam)
{
    if (m_inputHeight > 0 && m_inputWidth > 0)
    {
        transformParam.resize_wh = {m_inputWidth, m_inputHeight};
        if (m_keepRatio)
        {
            float x_factor = m_inputWidth * 1.0f / src.cols;
            float y_factor = m_inputHeight * 1.0f / src.rows;
            float factor = x_factor > y_factor ? y_factor : x_factor;
            transformParam.resize_wh = {int(src.cols * factor), int(src.rows * factor)};
            if (m_PadAfterResize)
            {
                int pad_h = m_inputHeight - transformParam.resize_wh.height;
                int pad_w = m_inputWidth - transformParam.resize_wh.width;
                transformParam.pad_left = pad_w / 2;
                transformParam.pad_right = pad_w - transformParam.pad_left;
                transformParam.pad_top = pad_h / 2;
                transformParam.pad_bottom = pad_h - transformParam.pad_top;
            }
        }
    }
    else
    {
        transformParam.resize_wh = {int(src.cols / m_downSampleBase * m_downSampleBase), int(src.rows / m_downSampleBase * m_downSampleBase)};
    }
    transformParam.origin_wh = {src.cols, src.rows};
    cv::resize(src, dst, transformParam.resize_wh, 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(dst, dst, transformParam.pad_top, transformParam.pad_bottom, transformParam.pad_left, transformParam.pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return true;
}

bool BaseInference::resizeInv(const TransformParam &transformParam, AiData::InnerModelOutput &output)
{
    float x_factor = transformParam.resize_wh.width * 1.0f / transformParam.origin_wh.width;
    float y_factor = transformParam.resize_wh.height * 1.0f / transformParam.origin_wh.height;

    auto it = output.det_result.boxes;
    int i = 0;
    while (i < output.det_result.n_boxes)
    {
        auto &x0 = *it++;
        auto &y0 = *it++;
        auto &x1 = *it++;
        auto &y1 = *it++;
        x0 = (x0 - transformParam.pad_left) / x_factor;
        x1 = (x1 - transformParam.pad_left) / x_factor;
        y0 = (y0 - transformParam.pad_top) / y_factor;
        y1 = (y1 - transformParam.pad_top) / y_factor;
        ++i;
    }
    return true;
}
