#include "CombinedPoseFace.h"
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <future>
#include "cpu.h"

const char* CombinedPoseFace::class_names_[2] = {"Person", "Face"};
const unsigned char CombinedPoseFace::colors_[10][3] = {
        {0, 0, 255},     // person (蓝色)
        {255, 0, 0},     // face (红色)
        {0, 255, 0},     // person_with_face (绿色)
        {99, 30, 233},   // bicycle (紫罗兰)
        {176, 39, 156},   // car (粉紫)
        {181, 81, 63},   // airplane (砖红)
        {243, 150, 33},  // bus (橙黄)
        {244, 169, 3},   // train (金黄)
        {212, 188, 0},   // truck (芥末黄)
        {136, 150, 0},   // boat (橄榄绿)
};
CombinedPoseFace::CombinedPoseFace(){ }
CombinedPoseFace::~CombinedPoseFace()
{
    PersonNet.clear();
    PoseNet.clear();
    FaceNet.clear();
}
ncnn::Mat CombinedPoseFace::preprocessImage(const cv::Mat& rgb, float& scale, int& wpad, int& hpad)
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
ncnn::Mat CombinedPoseFace::preprocessImage_face(const cv::Mat& rgb, float& scale, int& wpad, int& hpad)
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
// 添加快速指数函数，参考 DbFace 实现
inline float CombinedPoseFace::fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float CombinedPoseFace::myExp(float v) {
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
void CombinedPoseFace::genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, std::vector<Id_> &ids) {
    const float *ptr = hm.channel(0);
    const float *ptrPool = hmPool.channel(0);
    for (int i = 0; i < hm.w; i++) {
        float temp = 0.0;
        if ((ptr[i] - ptrPool[i]) < 0.01) {
            temp = ptr[i];
        }
        if (ptr[i] > thresh) {
            Id_ temp;
            temp.idx = i % w;
            temp.idy = (int) (i / w);
            temp.score = ptr[i];
            ids.push_back(temp);
        }
    }
}
void CombinedPoseFace::decode(int w, std::vector<Id_> ids, ncnn::Mat tlrb, std::vector<Obj_> &objs) {
    for (int i = 0; i < ids.size(); i++) {
        Obj_ objTemp;
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
        objTemp.box.x = (cx - boxTemp[0]) * 4;
        objTemp.box.y = (cy - boxTemp[1]) * 4;
        objTemp.box.r = (cx + boxTemp[2]) * 4;
        objTemp.box.b = (cy + boxTemp[3]) * 4;
        objTemp.score = score;
        // 清空关键点，避免访问未初始化的数据
        objTemp.keypoints.clear();
        objs.push_back(objTemp);
    }
}
float CombinedPoseFace::getIou(Box a, Box b) {
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
std::vector<Obj_> CombinedPoseFace::nms(std::vector<Obj_> objs, float iou) {
    if (objs.size() == 0) {
        return objs;
    }
    sort(objs.begin(), objs.end(), [](Obj_ a, Obj_ b) { return a.score < b.score; });
    std::vector<Obj_> keep;
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
int CombinedPoseFace::detectPersons(const cv::Mat& rgb, std::vector<cv::Rect>& personBoxes,float &prob_threshold, float &nms_threshold)
{
    personBoxes.clear();
    int width = rgb.cols;
    int height = rgb.rows;
    float scale;
    int wpad, hpad;
    ncnn::Mat input = preprocessImage(rgb, scale, wpad, hpad);
    input.substract_mean_normalize(mean_vals_1, norm_vals_1);
    // 人体检测推理
    ncnn::Extractor ex = PersonNet.create_extractor();
    ex.input("data", input);
    ncnn::Mat out;
    ex.extract("output", out);
    // 解析检测结果
    for (int i = 0; i < out.h; i++) {
        float x1, y1, x2, y2, score, label;
        const float *values = out.row(i);
        // 网络输出的坐标是在pad后的图像上的相对坐标
        x1 = values[2] * input.w;
        y1 = values[3] * input.h;
        x2 = values[4] * input.w;
        y2 = values[5] * input.h;
        // 坐标还原：减去pad偏移，然后除以缩放比例
        float original_x1 = (x1 - (wpad / 2)) / scale;
        float original_y1 = (y1 - (hpad / 2)) / scale;
        float original_x2 = (x2 - (wpad / 2)) / scale;
        float original_y2 = (y2 - (hpad / 2)) / scale;
        original_x1 = std::max(std::min(original_x1, (float)(width - 1)), 0.f);
        original_y1 = std::max(std::min(original_y1, (float)(height - 1)), 0.f);
        original_x2 = std::max(std::min(original_x2, (float)(width - 1)), 0.f);
        original_y2 = std::max(std::min(original_y2, (float)(height - 1)), 0.f);
        score = values[1];
        label = values[0];
        if (score >= prob_threshold) {
            cv::Rect originalPersonBox(original_x1, original_y1, original_x2 - original_x1, original_y2 - original_y1);
            personBoxes.push_back(originalPersonBox);
        }
    }
    return 0;
}
int CombinedPoseFace::detectPoseInPerson(const cv::Mat& rgb, const cv::Rect& personBox, std::vector<PoseKeyPoint>& keypoints,float &prob_threshold, float &nms_threshold)
{
    keypoints.clear();
    // 提取人体ROI
    cv::Mat roi = rgb(personBox).clone();
    int w = roi.cols;
    int h = roi.rows;
    // 预处理ROI
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                 roi.cols, roi.rows, pose_size_width, pose_size_height);
    in.substract_mean_normalize(mean_vals_2, norm_vals_2);

    // 姿态检测推理
    auto ex = PoseNet.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("hybridsequential0_conv7_fwd", out);
    // 解析关键点
    for (int p = 0; p < out.c; p++) {
        // 过滤掉面部关键点（索引0-4）
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
        if (max_prob >= prob_threshold) {
            PoseKeyPoint keypoint;
            keypoint.p = cv::Point2f(max_x * w / (float)out.w + personBox.x,
                                     max_y * h / (float)out.h + personBox.y);
            keypoint.prob = max_prob;
            keypoints.push_back(keypoint);
        }
    }
    return 0;
}

int CombinedPoseFace::detectFaces(const cv::Mat& rgb, std::vector<Object>& faceObjects,float &prob_threshold ,float &nms_threshold)
{
    faceObjects.clear();
    int width = rgb.cols;
    int height = rgb.rows;
    // 使用封装的预处理函数
    float scale;
    int wpad, hpad;
    ncnn::Mat input = preprocessImage_face(rgb, scale, wpad, hpad);

    input.substract_mean_normalize(mean_vals_3, norm_vals_3);
    ncnn::Extractor ex = FaceNet.create_extractor();
    ex.input("0", input);
    ncnn::Mat hm, hmPool, tlrb;
    //热力图，用于检测人脸中心位置
    ex.extract("hm", hm);
    //池化热力图，用于过滤候选点（非极大值抑制）
    ex.extract("pool_hm", hmPool);
    //边界框回归归（top, left, right, bottom），用于计算人脸检测框
    ex.extract("tlrb", tlrb);
    int hmWeight = hm.w;
    hm = hm.reshape(hm.c * hm.h * hm.w);
    hmPool = hmPool.reshape(hmPool.c * hmPool.w * hmPool.h);
    std::vector<Id_> ids;
    genIds(hm, hmPool, hmWeight, prob_threshold, ids);
    std::vector<Obj_> objs;
    decode(hmWeight, ids, tlrb, objs);
    std::vector<Obj_> nms_objs = nms(objs, (float)(1 - nms_threshold));
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
        object.Face_keyPoints.clear();
        faceObjects.push_back(object);
    }
    return 0;
}
int CombinedPoseFace::load(AAssetManager* mgr, int modelid, int inputsize, bool use_gpu)
{
    //0 高效  2 性能
    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    PersonNet.opt = ncnn::Option();
    PoseNet.opt = ncnn::Option();
    FaceNet.opt = ncnn::Option();
#if NCNN_VULKAN
    PersonNet.opt.use_vulkan_compute = use_gpu;
    PoseNet.opt.use_vulkan_compute = use_gpu;
    FaceNet.opt.use_vulkan_compute = use_gpu;
#endif
    // 启用轻量模式，减少内存使用
    PersonNet.opt.lightmode = true;
    PoseNet.opt.lightmode = true;
    FaceNet.opt.lightmode = true;

    const char* modeltype_1 = "PersonDetector";
    char parampath_1[256];
    char modelpath_1[256];
    sprintf(parampath_1, "%s.param", modeltype_1);
    sprintf(modelpath_1, "%s.bin", modeltype_1);
    PersonNet.load_param(mgr, parampath_1);
    PersonNet.load_model(mgr, modelpath_1);

    const char* modeltype_2 = "SimplePose";
    char parampath_2[256];
    char modelpath_2[256];
    sprintf(parampath_2, "%s.param", modeltype_2);
    sprintf(modelpath_2, "%s.bin", modeltype_2);
    PoseNet.load_param(mgr, parampath_2);
    PoseNet.load_model(mgr, modelpath_2);

    const char* modeltype_3 = "DbFace";
    char parampath_3[256];
    char modelpath_3[256];
    sprintf(parampath_3, "%s.param", modeltype_3);
    sprintf(modelpath_3, "%s.bin", modeltype_3);
    FaceNet.load_param(mgr, parampath_3);
    FaceNet.load_model(mgr, modelpath_3);
    target_size = (inputsize == 0) ? 320 : 640;
    return 0;
}
int CombinedPoseFace::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    float prob_threshold = g_threshold;
    float nms_threshold = g_nms;
    double t3 = ncnn::get_current_time();
    // 步骤1: 检测人体
    std::vector<cv::Rect> personBoxes;
    detectPersons(rgb, personBoxes,prob_threshold, nms_threshold);

    std::vector<Object> faceObjects;
    detectFaces(rgb, faceObjects, prob_threshold, nms_threshold);

    // 将人体检测结果直接添加到objects
    for (const auto& personBox : personBoxes) {
        Object personObj;
        personObj.rect = personBox;
        personObj.label = 0; // person
        personObj.prob = 0.8f; // 默认置信度

        cv::Rect expandedBox = personBox;
        float cx = personBox.x + personBox.width * 0.5f;
        float cy = personBox.y + personBox.height * 0.5f;
        float pw = personBox.width;
        float ph = personBox.height;
        expandedBox.x = cx - 0.7 * pw;
        expandedBox.y = cy - 0.6 * ph;
        expandedBox.width = 1.4 * pw;
        expandedBox.height = 1.2 * ph;
        // 边界检查
        expandedBox.x = std::max(0, expandedBox.x);
        expandedBox.y = std::max(0, expandedBox.y);
        expandedBox.width = std::min(expandedBox.width, rgb.cols - expandedBox.x);
        expandedBox.height = std::min(expandedBox.height, rgb.rows - expandedBox.y);

        // 确保ROI有效
        if (expandedBox.width > 0 && expandedBox.height > 0) {
            std::vector<PoseKeyPoint> keypoints;
            detectPoseInPerson(rgb, expandedBox, keypoints, prob_threshold, nms_threshold);
            personObj.keyPoints = keypoints; // 将姿态关键点添加到对应的人体对象中
        }
        objects.push_back(personObj);
    }
    // 将人脸检测结果直接追加到objects
    for (const auto& faceObj : faceObjects) {
        Object newFaceObj = faceObj;
        newFaceObj.label = 1; // face
        objects.push_back(newFaceObj);
    }
    double t4 = ncnn::get_current_time();
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.inferTimeMs = (t4 - t3);
    return 0;
}


