#ifndef SIMPLEPOSE_H
#define SIMPLEPOSE_H
#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>

#include "vision_base.h"
#include "IYoloAlgo.h"
// 人体姿态关键点定义：
// 0 nose, 1 left_eye, 2 right_eye, 3 left_Ear, 4 right_Ear (面部关键点 - 过滤掉)
// 5 left_Shoulder, 6 right_Shoulder, 7 left_Elbow, 8 right_Elbow, 9 left_Wrist, 10 right_Wrist
// 11 left_Hip, 12 right_Hip, 13 left_Knee, 14 right_Knee, 15 left_Ankle, 16 right_Ankle (身体关键点 - 保留)
class SimplePose: public IYoloAlgo
{
public:
    SimplePose();
    ~SimplePose() override;
    int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu = false)override;
    int detect(const cv::Mat& rgb, std::vector<Object>& objects)override;
    const char** getClassNames() const override { return class_names_; }
    int getClassCount() const override { return 2; }
    const unsigned char (*getColors() const)[3] override { return colors_; }
private:
    int runpose(cv::Mat &roi, int pose_size_width, int pose_size_height,
                std::vector<PoseKeyPoint> &keypoints,
                float x1, float y1);
    ncnn::Net PersonNet;
    ncnn::Net PoseNet;
    int target_size;
    static const char* class_names_[2];
    static const unsigned char colors_[10][3];
    const float mean_vals[3] =  {0, 0, 0};
    const float norm_vals[3] =  {1 / 255.f, 1 / 255.f, 1 / 255.f};
    const float mean_val[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_val[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    int pose_size_width = 192;
    int pose_size_height = 256;
    // 图像预处理函数：缩放和填充到32的倍数
    ncnn::Mat preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad);
};
#endif // SIMPLEPOSE_H
