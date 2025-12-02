#ifndef VISION_BASE_H
#define VISION_BASE_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>

#include <platform.h>
#include <benchmark.h>
#include <opencv2/core/core.hpp>

// 类别名称与颜色（由各算法实现文件提供）
extern const char* class_names[10];
extern const unsigned char colors[10][3];

// =============================
// 基础数据结构（检测结果 / 摘要）
// =============================

// 人体姿态关键点
struct PoseKeyPoint {
    cv::Point2f p;
    float prob;
};

// 人脸关键点
struct FaceKeyPoint {
    cv::Point2f p;
    float prob;
    int landmark_id;
};

// 分割结果
struct MarkPoint {
    cv::Mat mask;
    std::vector<float> mask_feat;
};

// 通用目标检测结果
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;

    // 分割结果
    MarkPoint markPoint;
    // 人体姿态
    std::vector<PoseKeyPoint> keyPoints;
    // 人脸关键点
    std::vector<FaceKeyPoint> Face_keyPoints;
};

// 检测摘要信息（提供给 Java 层）
struct DetectSummary {
    float allTimeMs;
    float inferTimeMs;
    float fps;
    std::string logText;                  // 目标统计文本
    std::vector<std::string> class_info;  // 各类别计数
};

// =============================
// 全局状态
// =============================

extern DetectSummary g_summary;
extern std::unique_ptr<DetectSummary> g_summary_cache;
extern std::mutex g_summary_mutex;

// ncnn 全局锁（保护 g_yolo / 推理流程）
extern ncnn::Mutex g_lock;

// 前向声明：算法基类
class IYoloAlgo;

// 当前加载的算法实例
extern IYoloAlgo* g_yolo;

// 推理配置
extern float g_threshold;
extern float g_nms;
extern bool trackEnabled;
extern bool shaderEnabled;

// =============================
// 与检测/绘制无关的统计与工具函数
// =============================

// 由检测结果统计各类别数量，生成日志与类别信息列表
std::pair<std::string, std::vector<std::string>>
updateDetectSummary(const std::vector<Object>& objects, const char** class_names);

// 生成检测结果日志字符串（仅统计，不做绘制）
std::string buildDetectLog(const std::vector<Object>& objects, const char* class_names[]);

// 将检测框与关键点从网络输入尺度还原到原图尺寸
void restoreObjectsToOriginal(std::vector<Object>& objects,
                              int padX, int padY, float scale,
                              int origW, int origH);

// 统计类别信息并返回日志字符串
std::string getClassCountLog(const std::vector<Object>& objects, const char* class_names[]);

#endif // VISION_BASE_H


