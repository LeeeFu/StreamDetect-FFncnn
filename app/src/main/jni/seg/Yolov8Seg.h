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

#ifndef YOLOSeg_H
#define YOLOSeg_H
#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>

#include "globals.h"
#include "IYoloAlgo.h"

namespace yolov8seg {
    struct GridAndStride{
        int grid0;
        int grid1;
        int stride;};
}
class Yolov8Seg: public IYoloAlgo
{
public:
    Yolov8Seg() ;
    ~Yolov8Seg() override;
    int load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu = false)override;
    int detect(const cv::Mat& rgb, std::vector<Object>& objects)override;
    const char** getClassNames() const override { return class_names_; }
    int getClassCount() const override { return 80; }
    const unsigned char (*getColors() const)[3] override { return colors_; }
private:
    ncnn::Net yoloseg;
    int target_size;
    static const char* class_names_[80];
    static const unsigned char colors_[80][3];
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    ncnn::Mat preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad);
};
#endif // Yolov8Seg
