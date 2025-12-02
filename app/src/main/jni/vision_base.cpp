#include "vision_base.h"

#include <map>
#include <cstring>

// 全局变量定义
DetectSummary g_summary;
std::unique_ptr<DetectSummary> g_summary_cache = nullptr;
std::mutex g_summary_mutex;
ncnn::Mutex g_lock;
IYoloAlgo* g_yolo = nullptr;
float g_threshold = 0.45f;
float g_nms = 0.65f;
bool trackEnabled = false;
bool shaderEnabled = false;

std::pair<std::string, std::vector<std::string>>
updateDetectSummary(const std::vector<Object>& objects, const char** class_names)
{
    std::map<int, int> cls_count;
    std::string logText;
    std::vector<std::string> classInfo;

    for (const auto& obj : objects)
        cls_count[obj.label]++;

    if (!cls_count.empty())
    {
        int totalTargets = 0;
        for (const auto& kv : cls_count)
            totalTargets += kv.second;

        logText += "Detect_Info: " + std::to_string(totalTargets) + " target, ";

        bool first = true;
        for (const auto& kv : cls_count)
        {
            if (!first)
                logText += ", ";

            logText += std::to_string(kv.second) + " " + class_names[kv.first];

            char buf[64];
            snprintf(buf, sizeof(buf), "%s: %d", class_names[kv.first], kv.second);
            classInfo.push_back(buf);

            first = false;
        }
    }

    return {logText, classInfo};
}

std::string buildDetectLog(const std::vector<Object>& objects, const char* class_names[])
{
    std::map<int, int> cls_count;
    for (const auto& obj : objects)
        cls_count[obj.label]++;

    std::string logLine;
    for (const auto& kv : cls_count)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s: %d", class_names[kv.first], kv.second);
        if (!logLine.empty())
            logLine += ", ";
        logLine += buf;
    }

    return logLine.empty() ? "None" : logLine;
}

void restoreObjectsToOriginal(std::vector<Object>& objects,
                              int padX, int padY, float scale,
                              int origW, int origH)
{
    for (auto& obj : objects)
    {
        obj.rect.x     = (obj.rect.x - padX) / scale;
        obj.rect.y     = (obj.rect.y - padY) / scale;
        obj.rect.width  /= scale;
        obj.rect.height /= scale;

        obj.rect.x = std::max(0.f, std::min(obj.rect.x, (float)(origW - 1)));
        obj.rect.y = std::max(0.f, std::min(obj.rect.y, (float)(origH - 1)));
        obj.rect.width = std::max(0.f, std::min(obj.rect.width, (float)(origW - obj.rect.x)));
        obj.rect.height = std::max(0.f, std::min(obj.rect.height, (float)(origH - obj.rect.y)));

        // 还原人体关键点坐标
        for (auto& kp : obj.keyPoints)
        {
            kp.p.x = (kp.p.x - padX) / scale;
            kp.p.y = (kp.p.y - padY) / scale;

            kp.p.x = std::max(0.f, std::min(kp.p.x, (float)(origW - 1)));
            kp.p.y = std::max(0.f, std::min(kp.p.y, (float)(origH - 1)));
        }

        // 还原人脸关键点坐标
        for (auto& kp : obj.Face_keyPoints)
        {
            kp.p.x = (kp.p.x - padX) / scale;
            kp.p.y = (kp.p.y - padY) / scale;

            kp.p.x = std::max(0.f, std::min(kp.p.x, (float)(origW - 1)));
            kp.p.y = std::max(0.f, std::min(kp.p.y, (float)(origH - 1)));
        }
    }
}

std::string getClassCountLog(const std::vector<Object>& objects, const char* class_names[])
{
    std::map<int, int> cls_count;
    for (const auto& obj : objects)
        cls_count[obj.label]++;

    std::string logLine;
    for (const auto& kv : cls_count)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s: %d", class_names[kv.first], kv.second);
        if (!logLine.empty())
            logLine += ", ";
        logLine += buf;
    }

    return logLine.empty() ? "None" : logLine;
}


