#ifndef DBFACE_H
#define DBFACE_H
#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>

#include "vision_base.h"
#include "IYoloAlgo.h"

struct Box {
    float x, y, r, b;
};
struct Id {
    double score;
    int idx;
    int idy;
};

struct Obj {
    double score;
    Box box;
    std::vector<FaceKeyPoint> keypoints;  // 使用FaceKeyPoint向量
};
class DbFace: public IYoloAlgo
{
public:
    DbFace();
    ~DbFace() override;
    int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu = false)override;
    int detect(const cv::Mat& rgb, std::vector<Object>& objects)override;
    const char** getClassNames() const override { return class_names_; }
    int getClassCount() const override { return 1; }
    const unsigned char (*getColors() const)[3] override { return colors_; }
private:
    ncnn::Net FaceNet;
    int target_size;
    float STRIDE = 4;
    static const char* class_names_[1];
    static const unsigned char colors_[10][3];
    const float mean_vals[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float norm_vals[3] = {1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f};
    ncnn::Mat preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad);
    void genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, std::vector<Id> &ids);
    void decode(int w, std::vector<Id> ids, ncnn::Mat tlrb, ncnn::Mat landmark, std::vector<Obj> &objs);
    inline float myExp(float v);
    std::vector<Obj> nms(std::vector<Obj> objs, float iou);
    inline float fast_exp(float x);
    float getIou(Box a, Box b);
};
#endif // DBFACE_H
