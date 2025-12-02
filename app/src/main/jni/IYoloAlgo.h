#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <android/asset_manager.h>
#include "vision_base.h" // Object结构体

class IYoloAlgo {
public:
    virtual ~IYoloAlgo() = 0;
    // 加载模型接口
    virtual int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu) = 0;
    // 推理接口
    virtual int detect(const cv::Mat& input, std::vector<Object>& objects) = 0;
    // 新增接口
    virtual const char** getClassNames() const = 0;
    virtual int getClassCount() const = 0;
    virtual const unsigned char (*getColors() const)[3] = 0;
};
inline IYoloAlgo::~IYoloAlgo() {}