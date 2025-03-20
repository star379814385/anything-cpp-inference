#include "pch.h"

namespace AiData{
    typedef struct DetResult
    {
        uint32_t n_boxes;
        float* boxes;   // [xyxy] * n
        float* scores; 
        int* labels;
    }DetResult;

    typedef struct InnerModelInput
    {
        cv::Mat img;
    }InnerModelInput;

    typedef struct InnerModelOutput
    {
        DetResult det_result;
    }InnerModelOutput;

};

typedef struct TransformParam
{
    cv::Size2i origin_wh;
    cv::Size2i resize_wh;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
}TransfromParam;