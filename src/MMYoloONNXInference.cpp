#include "MMYoloONNXInference.h"

// bool MMYoloONNXInference::postprocess(const std::vector<Ort::Value> &onnx_output, AiData::InnerModelOutput &output, TransformParam transformParam)
// {
//     auto num_dets = onnx_output[0].GetTensorData<int64_t>();
//     auto boxes = onnx_output[1].GetTensorData<float>();
//     auto scores = onnx_output[2].GetTensorData<float>();
//     auto labels = onnx_output[3].GetTensorData<int32_t>();

//     if (m_scoreThr == 0)
//     {
//         output.det_result.n_boxes = int(*num_dets);
//         output.det_result.boxes = (float *)boxes;
//         output.det_result.scores = (float *)scores;
//         output.det_result.labels = (int *)labels;
//     }
//     else
//     {
//     };
//     float x_factor = transformParam.resize_wh.width * 1.0f / transformParam.origin_wh.width;
//     float y_factor = transformParam.resize_wh.height * 1.0f / transformParam.origin_wh.height;

//     auto it = output.det_result.boxes;
//     int i = 0;
//     while (i < output.det_result.n_boxes)
//     {
//         auto &x0 = *it++;
//         auto &y0 = *it++;
//         auto &x1 = *it++;
//         auto &y1 = *it++;
//         x0 = (x0 - transformParam.pad_left) / x_factor;
//         x1 = (x1 - transformParam.pad_left) / x_factor;
//         y0 = (y0 - transformParam.pad_top) / y_factor;
//         y1 = (y1 - transformParam.pad_top) / y_factor;
//         ++i;
//     }

//     return true;
// }

// bool MMYoloONNXInference::postprocess(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs, std::vector<TransformParam> transform_param_list)
// {
//     auto num_dets = onnx_output[0].GetTensorData<int64_t>();
//     auto boxes = onnx_output[1].GetTensorData<float>();
//     auto scores = onnx_output[2].GetTensorData<float>();
//     auto labels = onnx_output[3].GetTensorData<int32_t>();

//     outputs.resize(transform_param_list.size());
//     for(size_t i = 0; i < outputs.size(); ++i)
//     {
//         auto &output = outputs[i];
//         auto cur_num_det = *(num_dets + i);
//         output.det_result.boxes = (float *)boxes;
//         output.det_result.scores = (float *)scores;
//         output.det_result.labels = (int *)labels;
//         boxes += cur_num_det * 4;
//         scores += cur_num_det;
//         labels += cur_num_det;
//         resizeInv(transform_param_list[i], output);
//     }

//     return true;
// }

bool MMYoloONNXInference::decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs)
{
    int batchsize = onnx_output[0].GetTensorTypeAndShapeInfo().GetShape()[0];
    // int batchsize = 1;
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
