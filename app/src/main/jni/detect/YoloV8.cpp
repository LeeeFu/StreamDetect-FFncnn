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
#include "YoloV8.h"
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "cpu.h"
const char* YoloV8::class_names_[80] = {"person", "bicycle", "car", "motorcycle", "airplane",
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
const unsigned char YoloV8::colors_[80][3] = {
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

        // 补齐剩余70个颜色（保持高饱和度风格）
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

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
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
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<yolov8::GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                yolov8::GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
static void generate_proposals(std::vector<yolov8::GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 80;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

YoloV8::YoloV8()
{
//    blob_pool_allocator.set_size_compare_ratio(0.f);
//    workspace_pool_allocator.set_size_compare_ratio(0.f);
}
YoloV8::~YoloV8()
{
    yolo.clear();
//    blob_pool_allocator.clear();
//    workspace_pool_allocator.clear();
}

int YoloV8::load(AAssetManager* mgr,  int modelid, int inputsize,bool use_gpu)
{
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    yolo.opt = ncnn::Option();
#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif
    yolo.opt.lightmode = true;
//    yolo.opt.blob_allocator = &blob_pool_allocator;
//    yolo.opt.workspace_allocator = &workspace_pool_allocator;
    const char* modeltype = (modelid == 1) ? "YoloV8n" : "YoloV8s";
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);
    target_size = (inputsize == 0) ? 320 : 640;
    return 0;
}
// 图像预处理函数：缩放和填充到32的倍数
ncnn::Mat YoloV8::preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad)
{
    int width = rgb.cols;
    int height = rgb.rows;
    // pad to multiple of 32
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
int YoloV8::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    float prob_threshold =g_threshold;
    float nms_threshold =g_nms;
    int width = rgb.cols;
    int height = rgb.rows;
    // 使用封装的预处理函数
    float scale;
    int wpad, hpad;
    ncnn::Mat in_pad = preprocessImage(rgb, scale, wpad, hpad);
    in_pad.substract_mean_normalize(0, norm_vals);
    double t3 = ncnn::get_current_time();
    ncnn::Extractor ex = yolo.create_extractor();
    ex.set_light_mode(true);
    ex.input("images", in_pad);
    std::vector<Object> proposals;
    ncnn::Mat out;
    ex.extract("output", out);  //add
    double t4 = ncnn::get_current_time();
    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<yolov8::GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals);
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.inferTimeMs = (t4 - t3); // 单位：秒
    return 0;
}