// #include "ONNXInference.h"

// class UltralyticsONNXInference: public ONNXInference
// {
// public:
//     UltralyticsONNXInference()
//     {
//         m_swapRb = true;
//         m_downSampleBase = 32;
//         m_keepRatio = false;
//         m_PadAfterResize = false;
//         m_mean[0] = 0.0f;
//         m_mean[1] = 0.0f;
//         m_mean[2] = 0.0f;
//         m_std[0] = 1.0f * 255;
//         m_std[1] = 1.0f * 255;
//         m_std[2] = 1.0f * 255;
//     };
//     ~UltralyticsONNXInference()
//     {

//     };
// protected:    
//     virtual bool update_from_config(const std::string modelDir)
//     {
//         ONNXInference::update_from_config(modelDir);
//         m_scoreThr = 0;        
//         return true;
//     };

// protected:
//     virtual bool postprocess(const std::vector<Ort::Value> &onnx_output, AiData::InnerModelOutput &output, TransformParam transformParam) override;
//     float m_scoreThr;

// };