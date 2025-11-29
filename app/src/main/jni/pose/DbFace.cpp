#include "DbFace.h"
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "cpu.h"
const char* DbFace::class_names_[1] = {"Face"};
const unsigned char DbFace::colors_[10][3] = {
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
DbFace::DbFace(){}
DbFace::~DbFace()
{
    FaceNet.clear();
}
void DbFace::genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, std::vector<Id> &ids) {
    const float *ptr = hm.channel(0);
    const float *ptrPool = hmPool.channel(0);
    for (int i = 0; i < hm.w; i++) {
        float temp = 0.0;
        if ((ptr[i] - ptrPool[i]) < 0.01) {
            temp = ptr[i];
        }
        if (ptr[i] > thresh) {
            Id temp;
            temp.idx = i % w;
            temp.idy = (int) (i / w);
            temp.score = ptr[i];
            ids.push_back(temp);
        }
    }
}
inline float DbFace::fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
inline float DbFace::myExp(float v) {
    float gate = 1;
    float base = exp(1);
    if (abs(v) < gate) {
        return v * base;
    }
    if (v > 0) {
        return fast_exp(v);
    } else {
        return -fast_exp(-v);
    }
}
float DbFace::getIou(Box a, Box b) {
    float aArea = (a.r - a.x + 1) * (a.b - a.y + 1);
    float bArea = (b.r - b.x + 1) * (b.b - b.y + 1);

    float x1 = a.x > b.x ? a.x : b.x;
    float y1 = a.y > b.y ? a.y : b.y;
    float x2 = a.r < b.r ? a.r : b.r;
    float y2 = a.b < b.b ? a.b : b.b;
    float w = 0.0f > x2 - x1 + 1 ? 0.0f : x2 - x1 + 1;
    float h = 0.0f > y2 - y1 + 1 ? 0.0f : y2 - y1 + 1;
    float area = w * h;

    float iou = area / (aArea + bArea - area);
    return iou;
}
void DbFace::decode(int w, std::vector<Id> ids, ncnn::Mat tlrb, ncnn::Mat landmark, std::vector<Obj> &objs) {
    for (int i = 0; i < ids.size(); i++) {
        Obj objTemp;
        int cx = ids[i].idx;
        int cy = ids[i].idy;
        double score = ids[i].score;
        // 添加边界检查
        if (cy <= 0 || cy > tlrb.h || cx < 0 || cx >= tlrb.w) {
            continue;
        }
        std::vector<float> boxTemp;
        for (int j = 0; j < tlrb.c; j++) {
            const float *ptr = tlrb.channel(j);
            boxTemp.push_back(ptr[w * (cy - 1) + cx]);
        }
        objTemp.box.x = (cx - boxTemp[0]) * STRIDE;
        objTemp.box.y = (cy - boxTemp[1]) * STRIDE;
        objTemp.box.r = (cx + boxTemp[2]) * STRIDE;
        objTemp.box.b = (cy + boxTemp[3]) * STRIDE;
        objTemp.score = score;
        objTemp.keypoints.clear();
        for (int j = 0; j < 10; j++) {
            const float *ptr = landmark.channel(j);
            if (j < 5) {
                float temp = (myExp(ptr[w * (cy - 1) + cx] * 4) + cx) * STRIDE;
                FaceKeyPoint kp;
                kp.p = cv::Point2f(temp, 0);
                kp.prob = 1.0f;
                kp.landmark_id = j;
                objTemp.keypoints.push_back(kp);
            } else {
                float temp = (myExp(ptr[w * (cy - 1) + cx] * 4) + cy) * STRIDE;
                // 更新y坐标
                if (j - 5 < objTemp.keypoints.size()) {
                    objTemp.keypoints[j - 5].p.y = temp;
                }
            }
        }
        objs.push_back(objTemp);
    }
}
std::vector<Obj> DbFace::nms(std::vector<Obj> objs, float iou) {
    if (objs.size() == 0) {
        return objs;
    }
    sort(objs.begin(), objs.end(), [](Obj a, Obj b) { return a.score < b.score; });
    std::vector<Obj> keep;
    int *flag = new int[objs.size()]();
    for (int i = 0; i < objs.size(); i++) {
        if (flag[i] != 0) {
            continue;
        }
        keep.push_back(objs[i]);
        for (int j = i + 1; j < objs.size(); j++) {
            if (flag[j] == 0 && getIou(objs[i].box, objs[j].box) > iou) {
                flag[j] = 1;
            }
        }
    }
    delete[] flag;
    return keep;
}
int DbFace::load(AAssetManager* mgr,  int modelid, int inputsize,bool use_gpu)
{
    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    FaceNet.opt = ncnn::Option();
#if NCNN_VULKAN
    FaceNet.opt.use_vulkan_compute = use_gpu;
#endif
    // 启用轻量模式，减少内存使用
    FaceNet.opt.lightmode = true;

    const char* modeltype = "DbFace";
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    FaceNet.load_param(mgr, parampath);
    FaceNet.load_model(mgr, modelpath);
    target_size = (inputsize == 0) ? 320 : 640;
    return 0;
}
ncnn::Mat DbFace::preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad)
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
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
    // pad to target_size rectangle
    wpad = (w + 31) / 32 * 32 - w;
    hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    return in_pad;
}
int DbFace::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    float prob_threshold = g_threshold;
    float nms_threshold = g_nms;
    int width = rgb.cols;
    int height = rgb.rows;
    // 使用封装的预处理函数
    float scale;
    int wpad, hpad;
    ncnn::Mat in_pad = preprocessImage(rgb, scale, wpad, hpad);
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    double t3 = ncnn::get_current_time();
    ncnn::Extractor ex = FaceNet.create_extractor();
    ex.input("0", in_pad);
    ncnn::Mat landmark, hm, hmPool, tlrb;
    ex.extract("landmark", landmark);
    ex.extract("hm", hm);
    ex.extract("pool_hm", hmPool);
    ex.extract("tlrb", tlrb);
    int hmWeight = hm.w;
    hm = hm.reshape(hm.c * hm.h * hm.w);
    hmPool = hmPool.reshape(hmPool.c * hmPool.w * hmPool.h);
    std::vector<Id> ids;
    genIds(hm, hmPool, hmWeight, prob_threshold, ids);
    std::vector<Obj> objs;
    decode(hmWeight, ids, tlrb, landmark, objs);
    std::vector<Obj> nms_objs = nms(objs, (float)(1 - nms_threshold));
    for (const auto& obj : nms_objs) {
        Object object;
        float x0 = (obj.box.x - (wpad / 2)) / scale;
        float y0 = (obj.box.y - (hpad / 2)) / scale;
        float x1 = (obj.box.r - (wpad / 2)) / scale;
        float y1 = (obj.box.b - (hpad / 2)) / scale;
        // 边界裁剪，确保坐标在有效范围内
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        object.rect.x = x0;
        object.rect.y = y0;
        object.rect.width = x1 - x0;
        object.rect.height = y1 - y0;
        // 设置置信度和标签
        object.prob = (float)obj.score;
        object.label = 0; // 人脸类别标签
        std::vector<FaceKeyPoint> restored_keypoints;
        for (const auto& kp : obj.keypoints) {
            FaceKeyPoint restored_kp;
            // 关键点坐标也需要同样的还原处理
            float kp_x = (kp.p.x - (wpad / 2)) / scale;
            float kp_y = (kp.p.y - (hpad / 2)) / scale;
            // 边界裁剪
            kp_x = std::max(std::min(kp_x, (float)(width - 1)), 0.f);
            kp_y = std::max(std::min(kp_y, (float)(height - 1)), 0.f);
            restored_kp.p = cv::Point2f(kp_x, kp_y);
            restored_kp.prob = kp.prob;
            restored_kp.landmark_id = kp.landmark_id;
            restored_keypoints.push_back(restored_kp);
        }
        object.Face_keyPoints = restored_keypoints;
        objects.push_back(object);
    }
    double t4 = ncnn::get_current_time();
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.inferTimeMs = (t4 - t3); // 单位：秒
    return 0;
}