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
#include "SimplePose.h"
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "cpu.h"
const char* SimplePose::class_names_[2] = {"person", "person"};
const unsigned char SimplePose::colors_[10][3] = {
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
SimplePose::SimplePose(){}
SimplePose::~SimplePose()
{
    PersonNet.clear();
    PoseNet.clear();
}
int SimplePose::runpose(cv::Mat &roi, int pose_size_w, int pose_size_h, std::vector<PoseKeyPoint> &keypoints,
                        float x1, float y1) {
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_RGB, \
                                                 roi.cols, roi.rows, pose_size_w, pose_size_h);
    //数据预处理
    in.substract_mean_normalize(mean_val, norm_val);
    auto ex = PoseNet.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("hybridsequential0_conv7_fwd", out);
    keypoints.clear();
    for (int p = 0; p < out.c; p++) {
        if (p < 5) {
            continue;
        }
        const ncnn::Mat m = out.channel(p);
        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++) {

            const float *ptr = m.row(y);
            for (int x = 0; x < out.w; x++) {
                float prob = ptr[x];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }
        float kp_x = max_x * w / (float) out.w + x1;
        float kp_y = max_y * h / (float) out.h + y1;
        PoseKeyPoint keypoint;
        keypoint.p = cv::Point2f(kp_x, kp_y);
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }
    return 0;
}
int SimplePose::load(AAssetManager* mgr,  int modelid, int inputsize,bool use_gpu)
{
    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    PersonNet.opt = ncnn::Option();
    PoseNet.opt = ncnn::Option();
#if NCNN_VULKAN
    PersonNet.opt.use_vulkan_compute = use_gpu;
    PoseNet.opt.use_vulkan_compute = use_gpu;
#endif
    PersonNet.opt.lightmode = true;
    PoseNet.opt.lightmode = true;
    const char* modeltype = "PersonDetector";
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    PersonNet.load_param(mgr, parampath);
    PersonNet.load_model(mgr, modelpath);

    const char* modeltype_ = "SimplePose";
    char parampath_[256];
    char modelpath_[256];
    sprintf(parampath_, "%s.param", modeltype_);
    sprintf(modelpath_, "%s.bin", modeltype_);
    PoseNet.load_param(mgr, parampath_);
    PoseNet.load_model(mgr, modelpath_);
    target_size = (inputsize == 0) ? 320 : 640;
    return 0;
}
// 图像预处理函数：缩放和填充到32的倍数
ncnn::Mat SimplePose::preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad)
{
    int width = rgb.cols;
    int height = rgb.rows;

    int w = width;
    int h = height;
    scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);
    // pad to target_size rectangle
    wpad = (w + 31) / 32 * 32 - w;
    hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    return in_pad;
}
int SimplePose::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    float prob_threshold =g_threshold;
    float nms_threshold =g_nms;
    int width = rgb.cols;
    int height = rgb.rows;
    // 使用封装的预处理函数
    float scale;
    int wpad, hpad;
    ncnn::Mat in_pad = preprocessImage(rgb, scale, wpad, hpad);
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    double t3 = ncnn::get_current_time();
    ncnn::Extractor ex = PersonNet.create_extractor();
    ex.input("data", in_pad);
    ncnn::Mat out;
    ex.extract("output", out);
    for (int i = 0; i < out.h; i++) {
        Object obj;
        float x1, y1, x2, y2, score, label;
        float pw, ph, cx, cy;
        const float *values = out.row(i);
        // 网络输出的坐标是在pad后的图像上的相对坐标
        x1 = values[2] * in_pad.w;
        y1 = values[3] * in_pad.h;
        x2 = values[4] * in_pad.w;
        y2 = values[5] * in_pad.h;
        // 坐标还原：减去pad偏移，然后除以缩放比例
        x1 = (x1 - (wpad / 2)) / scale;
        y1 = (y1 - (hpad / 2)) / scale;
        x2 = (x2 - (wpad / 2)) / scale;
        y2 = (y2 - (hpad / 2)) / scale;

        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = x2 - x1;
        obj.rect.height = y2 - y1;

        pw = x2 - x1;
        ph = y2 - y1;
        cx = x1 + 0.5 * pw;
        cy = y1 + 0.5 * ph;

        x1 = cx - 0.7 * pw;
        y1 = cy - 0.6 * ph;
        x2 = cx + 0.7 * pw;
        y2 = cy + 0.6 * ph;

        score = values[1];
        label = values[0];

        // 边界裁剪，确保坐标在有效范围内
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(width - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(height - 1)), 0.f);
        cv::Mat roi = rgb(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        std::vector<PoseKeyPoint> keypoints;
        runpose(roi, pose_size_width, pose_size_height, keypoints, x1, y1);

        obj.label = label;
        obj.prob = score;
        obj.keyPoints = keypoints;
        objects.push_back(obj);
    }
    double t4 = ncnn::get_current_time();
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.inferTimeMs = (t4 - t3); // 单位：秒
    return 0;
}