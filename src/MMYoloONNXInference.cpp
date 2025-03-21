#include "MMYoloONNXInference.h"

bool MMYoloONNXInference::decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs)
{
    int batchsize = onnx_output[0].GetTensorTypeAndShapeInfo().GetShape()[0];
    outputs.resize(batchsize);

    auto num_dets = onnx_output[0].GetTensorData<int64_t>();
    auto boxes = onnx_output[1].GetTensorData<float>();
    auto scores = onnx_output[2].GetTensorData<float>();
    auto labels = onnx_output[3].GetTensorData<int32_t>();

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto &output = outputs[i];
        auto cur_num_det = *(num_dets + i);
        output.det_result.n_boxes = cur_num_det;
        output.det_result.boxes = (float *)boxes;
        output.det_result.scores = (float *)scores;
        output.det_result.labels = (int *)labels;
        boxes += cur_num_det * 4;
        scores += cur_num_det;
        labels += cur_num_det;
    }
    return true;
}
