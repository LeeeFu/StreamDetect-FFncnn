#ifndef NANODET_H
#define NANODET_H
#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>
#include "globals.h"
#include "IYoloAlgo.h"
typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};
class NanoDet: public IYoloAlgo
{
public:
    NanoDet();
    ~NanoDet() override;
    int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu = false)override;
    int detect(const cv::Mat& rgb, std::vector<Object>& objects)override;
    const char** getClassNames() const override { return class_names_; }
    int getClassCount() const override { return 80; }
    const unsigned char (*getColors() const)[3] override { return colors_; }
private:
    Object disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride, float width_ratio, float height_ratio);
    void decode_infer(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, float threshold, std::vector<Object>& objects, float width_ratio, float height_ratio);
    ncnn::Net Nano_net;
    int target_size;
    int num_class = 80;
    static const char* class_names_[80];
    static const unsigned char colors_[80][3];
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.01712475f};
    std::vector<HeadInfo> heads_info{
            {"792", "795", 8},
            {"814", "817", 16},
            {"836", "839", 32},
    };
};

#endif // NANODET_H
