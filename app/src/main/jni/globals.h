#ifndef GLOBALS_H
#define GLOBALS_H
#include <vector>
#include <string>
#include <platform.h>
#include <benchmark.h>
#include <opencv2/core/core.hpp>
#include <memory>

extern const char* class_names[10];
extern const unsigned char colors[10][3];
//人体姿态
struct PoseKeyPoint {
    cv::Point2f p;
    float prob;
};
//人脸
struct FaceKeyPoint {
    cv::Point2f p;
    float prob;
    int landmark_id;
};
// 分割结构体
struct MarkPoint {
    cv::Mat mask;
    std::vector<float> mask_feat;
};
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    //分割专用
    MarkPoint markPoint;
    //人体姿态专用
    std::vector<PoseKeyPoint> keyPoints;
    //人脸检测专用
    std::vector<FaceKeyPoint > Face_keyPoints;
};
struct DetectSummary {
    float allTimeMs;
    float inferTimeMs;
    float fps;
    float cpuUsage; // 新增字段，单位百分比
    std::string logText; // 只包含目标内容
    std::vector<std::string> class_info;
};
// 全局变量声明
extern DetectSummary g_summary;
extern std::unique_ptr<DetectSummary> g_summary_cache;
extern std::mutex g_summary_mutex;
extern ncnn::Mutex g_lock;
// 前向声明
class IYoloAlgo;
extern IYoloAlgo* g_yolo;
extern float g_threshold;
extern float g_nms;
extern bool trackEnabled;
extern bool shaderEnabled;

// 公共函数声明
std::pair<std::string, std::vector<std::string>> updateDetectSummary(const std::vector<Object>& objects, const char** class_names);
float get_cpu_usage();   // CPU利用率采集
int get_cpu_cores();     // 获取CPU核心数
#endif // GLOBALS_H
