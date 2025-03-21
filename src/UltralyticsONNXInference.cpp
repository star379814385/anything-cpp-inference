#include "UltralyticsONNXInference.h"

bool UltralyticsONNXInference::decode_result(const std::vector<Ort::Value> &onnx_output, std::vector<AiData::InnerModelOutput> &outputs)
{
    float rectConfidenceThreshold = m_scoreThr;
    float iouThreshold = m_iouThr;

    auto &outputTensor = onnx_output;
	auto tensor_info = outputTensor[0].GetTensorTypeAndShapeInfo();
	std::vector<int64_t>outputNodeDims = tensor_info.GetShape();
	auto output = (float*)outputTensor[0].GetTensorData<float>();

    int batchsize = outputNodeDims[0];
    int signalResultNum = outputNodeDims[1];    
    int strideNum = outputNodeDims[2];
    int classesNum = signalResultNum - 4;

    outputs.resize(batchsize);

    for(size_t batch_i = 0; batch_i < batchsize; ++batch_i)
    {
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect2d> boxes;
        cv::Mat rowData(signalResultNum, strideNum, CV_32F, output + batch_i * signalResultNum * strideNum);
        rowData = rowData.t();

        float* data = (float*)rowData.data;

        for (int i = 0; i < strideNum; ++i)
        {
            float* classesScores = data + 4;
            cv::Mat scores(1, classesNum, CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                double x = data[0];
                double y = data[1];
                double w = data[2];
                double h = data[3];

                double left = (x - 0.5 * w);
                double top = (y - 0.5 * h);

                boxes.push_back(cv::Rect2d(left, top, w, h));
            }
            data += signalResultNum;
        }

        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        outputs[batch_i].det_result.n_boxes = nmsResult.size();
        outputs[batch_i].det_result.labels = new int[nmsResult.size()];
        outputs[batch_i].det_result.boxes = new float[nmsResult.size() * 4];
        outputs[batch_i].det_result.scores = new float[nmsResult.size()];
        
        for (int i = 0; i < nmsResult.size(); ++i)
        {

            int idx = nmsResult[i];
            *(outputs[batch_i].det_result.labels + i) = class_ids[idx];
            *(outputs[batch_i].det_result.scores + i) = confidences[idx];
            float* it = outputs[batch_i].det_result.boxes + i * 4;
            *it = float(boxes[idx].x);
            *(it + 1) = float(boxes[idx].y);
            *(it + 2) = float(boxes[idx].x + boxes[idx].width);
            *(it + 3) = float(boxes[idx].y + boxes[idx].height);
        }
    }
    return true;

}