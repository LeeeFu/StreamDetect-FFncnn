#ifndef COMBINEDPOSEFACE_H
#define COMBINEDPOSEFACE_H
#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>

#include "vision_base.h"
#include "IYoloAlgo.h"
#include "SimplePose.h"
#include "DbFace.h"
struct Box_ {
    float x, y, r, b;
};
struct Id_ {
    double score;
    int idx;
    int idy;
};
struct Obj_ {
    double score;
    Box box;
    std::vector<FaceKeyPoint> keypoints;  // 使用FaceKeyPoint向量
};
class CombinedPoseFace: public IYoloAlgo
{
public:
    CombinedPoseFace();
    ~CombinedPoseFace() override;
    int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu = false)override;
    int detect(const cv::Mat& rgb, std::vector<Object>& objects)override;
    const char** getClassNames() const override { return class_names_; }
    int getClassCount() const override { return 2; }
    const unsigned char (*getColors() const)[3] override { return colors_; }

private:
    ncnn::Net PersonNet;
    ncnn::Net PoseNet;
    ncnn::Net FaceNet;
    int target_size;
    inline float fast_exp(float x);
    inline float myExp(float v);
    // 三个独立的检测步骤
    int detectPersons(const cv::Mat& rgb, std::vector<cv::Rect>& personBoxes,float &prob_threshold ,float &nms_threshold);
    int detectPoseInPerson(const cv::Mat& rgb, const cv::Rect& personBox, std::vector<PoseKeyPoint>& keypoints,float &prob_threshold ,float &nms_threshold);
    int detectFaces(const cv::Mat& rgb, std::vector<Object>& faceObjects,float &prob_threshold ,float &nms_threshold);
    ncnn::Mat preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad);
    ncnn::Mat preprocessImage_face(const cv::Mat& rgb, float& scale, int& wpad, int& hpad);
    void genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, std::vector<Id_> &ids);
    void decode(int w, std::vector<Id_> ids, ncnn::Mat tlrb, std::vector<Obj_> &objs);
    std::vector<Obj_> nms(std::vector<Obj_> objs, float iou);
    float getIou(Box a, Box b);

    static const char* class_names_[2];
    static const unsigned char colors_[10][3];

    int pose_size_width = 192;
    int pose_size_height = 256;

    const float mean_vals_1[3] =  {0, 0, 0};
    const float norm_vals_1[3] =  {1 / 255.f, 1 / 255.f, 1 / 255.f};
    const float mean_vals_2[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals_2[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    const float mean_vals_3[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float norm_vals_3[3] = {1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f};
};

#endif // COMBINEDPOSEFACE_H