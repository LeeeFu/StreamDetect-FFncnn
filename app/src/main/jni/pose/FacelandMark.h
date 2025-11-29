#ifndef FACELANDMARK_H
#define FACELANDMARK_H
#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>

#include "globals.h"
#include "IYoloAlgo.h"
//if (landmark_id >= 0 && landmark_id <= 31) return "contour";
//if (landmark_id >= 32 && landmark_id <= 51) return "eyebrow";
//if (landmark_id >= 52 && landmark_id <= 71) return "nose";
//if (landmark_id >= 72 && landmark_id <= 95) return "eye";
//if (landmark_id >= 96 && landmark_id <= 105) return "mouth";
class FacelandMark: public IYoloAlgo
{
public:
    FacelandMark();
    ~FacelandMark() override;
    int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu = false)override;
    int detect(const cv::Mat& rgb, std::vector<Object>& objects)override;
    const char** getClassNames() const override { return class_names_; }
    int getClassCount() const override { return 1; }
    const unsigned char (*getColors() const)[3] override { return colors_; }
private:
    int runlandmark(cv::Mat &roi, int face_size_w, int face_size_h,
                    std::vector<FaceKeyPoint> &keypoints,
                    float x1, float y1);
    ncnn::Net FaceNet;
    ncnn::Net LandmarkNet;
    int target_size;
    static const char* class_names_[1];
    static const unsigned char colors_[10][3];
    const float mean_vals[3] =  {0, 0, 0};
    const float norm_vals[3] =  {1 / 255.f, 1 / 255.f, 1 / 255.f};
    const float mean_val[3] ={127.5f, 127.5f, 127.5f};
    const float norm_val[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    int detector_size_width = 320;
    int detector_size_height = 256;
    int landmark_size_width = 112;
    int landmark_size_height = 112;
};
#endif // FACELANDMARK_H
