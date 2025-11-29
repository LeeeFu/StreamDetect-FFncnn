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
#include "NanoDet.h"
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "cpu.h"
const char* NanoDet::class_names_[80] = {"person", "bicycle", "car", "motorcycle", "airplane",
                    "bus", "train", "truck", "boat", "traffic light",
                    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                    "cat", "dog", "horse", "sheep", "cow",
                    "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee",
                    "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                    "wine glass", "cup", "fork", "knife", "spoon",
                    "bowl", "banana", "apple", "sandwich", "orange",
                    "broccoli", "carrot", "hot dog", "pizza", "donut",
                    "cake", "chair", "couch", "potted plant", "bed",
                    "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven",
                    "toaster", "sink", "refrigerator", "book", "clock",
                    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
const unsigned char NanoDet::colors_[80][3] = {
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

        {255, 0, 0},     // fire hydrant (纯红)
        {0, 255, 255},    // stop sign (青色)
        {255, 0, 255},    // parking meter (洋红)
        {128, 0, 128},    // bench (深紫)
        {0, 128, 128},    // bird (蓝绿)
        {128, 128, 0},    // cat (土黄)
        {64, 224, 208},   // dog (绿松石)
        {210, 105, 30},   // horse (巧克力色)
        {218, 112, 214},  // sheep (兰花粉)
        {50, 205, 50},    // cow (酸橙绿)

        {255, 165, 0},    // elephant (橙色)
        {139, 69, 19},    // bear (棕褐)
        {220, 20, 60},    // zebra (猩红)
        {75, 0, 130},     // giraffe (靛蓝)
        {255, 192, 203},  // backpack (粉红)
        {173, 255, 47},   // umbrella (绿黄)
        {240, 230, 140},  // handbag (卡其)
        {245, 222, 179},  // tie (小麦色)
        {255, 228, 196},  // suitcase (杏仁白)
        {0, 139, 139},    // frisbee (深青)

        {148, 0, 211},    // skis (深紫罗兰)
        {255, 69, 0},     // snowboard (红橙)
        {154, 205, 50},   // sports ball (黄绿)
        {72, 209, 204},   // kite (中绿松石)
        {123, 104, 238},  // baseball bat (中紫)
        {106, 90, 205},   // baseball glove (中蓝紫)
        {60, 179, 113},   // skateboard (中海绿)
        {238, 130, 238},  // surfboard (紫罗兰)
        {255, 215, 0},    // tennis racket (金色)
        {34, 139, 34},    // bottle (森林绿)

        {219, 112, 147},  // wine glass (苍紫红)
        {46, 139, 87},    // cup (海绿)
        {112, 128, 144},  // fork (板岩灰)
        {47, 79, 79},     // knife (暗青灰)
        {188, 143, 143},  // spoon (玫棕)
        {255, 228, 225},  // bowl (雾玫瑰)
        {250, 128, 114},  // banana (鲜珊瑚)
        {216, 191, 216},  // apple (蓟色)
        {255, 222, 173},  // sandwich ( navajo白)
        {189, 183, 107},  // orange (暗卡其)

        {85, 107, 47},    // broccoli (暗橄榄绿)
        {107, 142, 35},   // carrot (橄榄褐)
        {152, 251, 152},  // hot dog (浅海绿)
        {205, 92, 92},    // pizza (印度红)
        {255, 160, 122},  // donut (鲜肉色)
        {139, 0, 0},      // cake (暗红)
        {233, 150, 122},  // chair (深肉色)
        {143, 188, 143},  // couch (暗海绿)
        {193, 205, 193},  // potted plant (薄荷霜)
        {240, 128, 128},  // bed (浅珊瑚红)

        {102, 205, 170},  // dining table (中海蓝绿)
        {151, 255, 255},  // toilet (天蓝)
        {255, 105, 180},  // tv (热粉红)
        {221, 160, 221},  // laptop (浅紫)
        {224, 255, 255},  // mouse (浅天蓝)
        {250, 250, 210},  // remote (浅黄)
        {144, 238, 144},  // keyboard (浅绿)
        {255, 182, 193},  // cell phone (浅粉红)
        {255, 255, 0},    // microwave (纯黄)
        {210, 180, 140},  // oven (棕褐色)

        {255, 20, 147},   // toaster (深粉红)
        {199, 21, 133},   // sink (中紫红)
        {25, 25, 112},    // refrigerator (午夜蓝)
        {230, 230, 250},  // book (薰衣草)
        {255, 248, 220},  // clock (玉米丝)
        {255, 250, 205},  // vase (柠檬绸)
        {139, 131, 120},  // scissors (暗灰)
        {0, 0, 128},      // teddy bear (海军蓝)
        {72, 61, 139},    // hair drier (暗蓝紫)
        {255, 99, 71}     // toothbrush (番茄红)
};
static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}
static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}
NanoDet::NanoDet(){

}
NanoDet::~NanoDet()
{
    Nano_net.clear();
}
void activation_function_softmax(const float* src, float* dst, int length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator = 0.0f;
    for (int i = 0; i < length; i++) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; i++) {
        dst[i] /= denominator;
    }
}
Object NanoDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride,
                             float width_ratio, float height_ratio)
{
    float ct_x = (x + 0.5f) * stride;
    float ct_y = (y + 0.5f) * stride;

    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float* dis_after_sm = new float[8]; // reg_max + 1 = 8
        activation_function_softmax(dfl_det + i * 8, dis_after_sm, 8);
        for (int j = 0; j < 8; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;  // 先乘以stride，再赋值
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = std::max(ct_x - dis_pred[0], 0.0f) * width_ratio;
    float ymin = std::max(ct_y - dis_pred[1], 0.0f) * height_ratio;
    float xmax = std::min(ct_x + dis_pred[2], (float)target_size) * width_ratio;
    float ymax = std::min(ct_y + dis_pred[3], (float)target_size) * height_ratio;

    Object obj;
    obj.rect.x = xmin;
    obj.rect.y = ymin;
    obj.rect.width = xmax - xmin;
    obj.rect.height = ymax - ymin;
    obj.label = label;
    obj.prob = score;
    return obj;
}

void NanoDet::decode_infer(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, float threshold,
                           std::vector<Object>& objects, float width_ratio, float height_ratio)
{
    int feature_h = target_size / stride;
    int feature_w = target_size / stride;

    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        const float* scores = cls_pred.row(idx);
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            const float* bbox_pred = dis_pred.row(idx);
            Object obj = disPred2Bbox(bbox_pred, cur_label, score, col, row, stride, width_ratio, height_ratio);
            objects.push_back(obj);
        }
    }
}

int NanoDet::load(AAssetManager* mgr,  int modelid, int inputsize,bool use_gpu)
{
    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    Nano_net.opt = ncnn::Option();
#if NCNN_VULKAN
    Nano_net.opt.use_vulkan_compute = use_gpu;
#endif
    Nano_net.opt.lightmode = true;
    const char* modeltype = "NanoDet";
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    Nano_net.load_param(mgr, parampath);
    Nano_net.load_model(mgr, modelpath);
    target_size = (inputsize == 0) ? 320 : 640;
    return 0;
}
int NanoDet::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    float prob_threshold =g_threshold;
    float nms_threshold =g_nms;
    int width = rgb.cols;
    int height = rgb.rows;
    //no padding
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, target_size, target_size);
    input.substract_mean_normalize(mean_vals, norm_vals);
    double t3 = ncnn::get_current_time();
    ncnn::Extractor ex = Nano_net.create_extractor();
    ex.input("input.1", input);
    std::vector<Object> proposals;
    for (const auto &head_info : this->heads_info) {
        ncnn::Mat dis_pred, cls_pred;
        ex.extract(head_info.dis_layer.c_str(), dis_pred);
        ex.extract(head_info.cls_layer.c_str(), cls_pred);
        decode_infer(cls_pred, dis_pred, head_info.stride, prob_threshold, proposals, float(width) / target_size, float(height) / target_size);
    }
    double t4 = ncnn::get_current_time();
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    objects.resize(picked.size());
    for (int i = 0; i < picked.size(); i++)
    {
        objects[i] = proposals[picked[i]];
    }
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.inferTimeMs = (t4 - t3); // 单位：秒
    return 0;
}