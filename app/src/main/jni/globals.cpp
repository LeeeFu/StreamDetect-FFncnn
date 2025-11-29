#include "globals.h"
#include <map>
#include <cstring>
#include <vector>
#include <string>
#include <mutex>
#include <fstream>
#include <sstream>

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
// 获取CPU核心数
int get_cpu_cores() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    int cores = 0;
    while (std::getline(file, line)) {
        if (line.find("processor") == 0) {
            cores++;
        }
    }
    return cores > 0 ? cores : 1; // 至少返回1
}
float get_cpu_usage() {
    static long lastTotalUser=0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
    static int cpuCores = 0;
    static bool firstCall = true;
    // 首次调用时获取CPU核心数
    if (cpuCores == 0) {
        cpuCores = get_cpu_cores();
    }
    std::ifstream file("/proc/stat");
    std::string line;
    if (!file.is_open()) return 0.f;
    std::getline(file, line);
    std::istringstream ss(line);
    std::string cpu;
    long user, nice, system, idle, iowait, irq, softirq, steal;
    ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    long totalUser = user;
    long totalUserLow = nice;
    long totalSys = system;
    long totalIdle = idle + iowait;
    float percent = 0.f;
    // 修复：首次调用时初始化基准值，不计算百分比
    if (firstCall) {
        lastTotalUser = totalUser;
        lastTotalUserLow = totalUserLow;
        lastTotalSys = totalSys;
        lastTotalIdle = totalIdle;
        firstCall = false;
        return 0.f; // 首次调用返回0
    }

    long total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
    long totalAll = total + (totalIdle - lastTotalIdle);
    if (totalAll > 0) {
        percent = 100.f * total / totalAll;
        // 转换为多核显示：单核100% = 多核100% * 核心数
        percent = percent * cpuCores;
    }
    // 更新基准值
    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;
    return percent;
}
std::pair<std::string, std::vector<std::string>> updateDetectSummary(const std::vector<Object>& objects, const char** class_names) {
    std::map<int, int> cls_count;
    std::string logText;
    std::vector<std::string> classInfo;
    for (const auto& obj : objects) cls_count[obj.label]++;
    if (!cls_count.empty()) {
        int totalTargets = 0;
        for (const auto &kv: cls_count) totalTargets += kv.second;
        logText += "Detect_Info: " + std::to_string(totalTargets) + " target, ";
        bool first = true;
        for (const auto &kv: cls_count) {
            if (!first) logText += ", ";
            logText += std::to_string(kv.second) + " " + class_names[kv.first];
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: %d", class_names[kv.first], kv.second);
            classInfo.push_back(buf);
            first = false;
        }
    }
    return {logText, classInfo};
}