#include "TimmONNXInference.h"

bool TimmONNXInference::decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs)
{
    auto tensor_shape = onnx_output[0].GetTensorTypeAndShapeInfo().GetShape();
    int batchsize = tensor_shape[0];
    int num_classes = tensor_shape[1];
    outputs.resize(batchsize);

    auto scores = onnx_output[0].GetTensorData<float>();

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto it = scores + num_classes * i;
        int idx = 0;
        float max_score = *it;
        for (size_t j = 0; j < num_classes; ++j)
        {
            if (*it > max_score)
            {
                max_score = *it;
                idx = j;
            }
            ++it;
        }
        std::cout << max_score << std::endl;
        outputs[i].cls_result.label_id = idx;
        // strcat(outputs[i].cls_result.label_name,  std::to_string(idx).c_str());
    }
    return true;
}