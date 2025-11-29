// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include "FacelandMark.h"
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "cpu.h"
const char* FacelandMark::class_names_[1] = {"Face"};
const unsigned char FacelandMark::colors_[10][3] = {
        {0, 0, 255},     // person (蓝色)
        {99, 30, 233},   // bicycle (紫罗兰)
        {176, 39, 156},   // car (粉紫)
        {0, 255, 0},     // motorcycle (纯绿)
        {181, 81, 63},   // airplane (砖红)
        {243, 150, 33},  // bus (橙黄)
        {244, 169, 3},   // train (金黄)
        {212, 188, 0},   // truck (芥末黄)
        {136, 150, 0},   // boat (橄榄绿)
        {80, 175, 76},   // traffic light (叶绿)
};
FacelandMark::FacelandMark()
{}
FacelandMark::~FacelandMark()
{
    FaceNet.clear();
    LandmarkNet.clear();
}
int FacelandMark::runlandmark(cv::Mat &roi, int face_size_w, int face_size_h, std::vector<FaceKeyPoint> &keypoints,
                        float x1, float y1) {
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_RGB2BGR, \
                                                 roi.cols, roi.rows, face_size_w, face_size_h);
    //数据预处理
    in.substract_mean_normalize(mean_val, norm_val);
    auto ex = LandmarkNet.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("bn6_3_bn6_3_scale", out);
    keypoints.clear();
    float sw, sh;
    sw = (float) w / (float) landmark_size_width;
    sh = (float) h / (float) landmark_size_height;
    for (int i = 0; i < 106; i++) {
        float px, py;
        px = out[i * 2] * landmark_size_width * sw + x1;
        py = out[i * 2 + 1] * landmark_size_height * sh + y1;
        FaceKeyPoint keypoint;
        keypoint.p = cv::Point2f(px, py);
        // 部位编号映射
        if (i >= 0 && i <= 31)
            keypoint.landmark_id = 0; // 轮廓
        else if (i >= 32 && i <= 51)
            keypoint.landmark_id = 1; // 眉毛
        else if (i >= 52 && i <= 71)
            keypoint.landmark_id = 2; // 鼻子
        else if (i >= 72 && i <= 95)
            keypoint.landmark_id = 3; // 眼睛
        else if (i >= 96 && i <= 105)
            keypoint.landmark_id = 4; // 嘴巴
        keypoints.push_back(keypoint);
    }
    return 0;
}
int FacelandMark::load(AAssetManager* mgr,  int modelid, int inputsize,bool use_gpu)
{
    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    FaceNet.opt = ncnn::Option();
    LandmarkNet.opt = ncnn::Option();
#if NCNN_VULKAN
    FaceNet.opt.use_vulkan_compute = use_gpu;
    LandmarkNet.opt.use_vulkan_compute = use_gpu;
#endif
    // 启用轻量模式，减少内存使用
    FaceNet.opt.lightmode = true;
    LandmarkNet.opt.lightmode = true;

    const char* modeltype = "YoloFace-500k";
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    FaceNet.load_param(mgr, parampath);
    FaceNet.load_model(mgr, modelpath);

    const char* modeltype_ = "LandMark106";
    char parampath_[256];
    char modelpath_[256];
    sprintf(parampath_, "%s.param", modeltype_);
    sprintf(modelpath_, "%s.bin", modeltype_);
    LandmarkNet.load_param(mgr, parampath_);
    LandmarkNet.load_model(mgr, modelpath_);
    return 0;
}
int FacelandMark::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    float prob_threshold =g_threshold;
    float nms_threshold =g_nms;
    int width = rgb.cols;
    int height = rgb.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,\
                                                 width, height, detector_size_width, detector_size_height);
    in.substract_mean_normalize(mean_vals, norm_vals);
    double t3 = ncnn::get_current_time();
    ncnn::Extractor ex = FaceNet.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);
    for (int i = 0; i < out.h; i++) {
        float x1, y1, x2, y2, score, label;
        float pw, ph, cx, cy;
        const float *values = out.row(i);
        x1 = values[2] * width;
        y1 = values[3] * height;
        x2 = values[4] * width;
        y2 = values[5] * height;

        pw = x2 - x1;
        ph = y2 - y1;
        cx = x1 + 0.5 * pw;
        cy = y1 + 0.5 * ph;

        x1 = cx - 0.55 * pw;
        y1 = cy - 0.35 * ph;
        x2 = cx + 0.55 * pw;
        y2 = cy + 0.55 * ph;

        score = values[1];
        label = values[0];

        // 边界裁剪，确保坐标在有效范围内
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(width - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(height - 1)), 0.f);

        Object obj;
        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = x2 - x1;
        obj.rect.height = y2 - y1;
        obj.label = 0;
        obj.prob = score;
        //截取脸ROI
        if (x2 - x1 > 66 && y2 - y1 > 66) {
            std::vector<FaceKeyPoint> keypoints;
            cv::Mat roi = rgb(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
            runlandmark(roi, landmark_size_width, landmark_size_height, keypoints, x1, y1);
            obj.Face_keyPoints = keypoints;
        }
        objects.push_back(obj);
    }
    double t4 = ncnn::get_current_time();
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.inferTimeMs = (t4 - t3); // 单位：秒
    return 0;
}