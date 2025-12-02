#ifndef VISION_INFER_H
#define VISION_INFER_H

#include <jni.h>
#include <vector>

#include <opencv2/core/core.hpp>

#include "vision_base.h"
#include "BYTETracker.h"

// 前向声明算法接口
class IYoloAlgo;

// =============================
// JNI / OpenCV 相关基础转换
// =============================

// Mat 转 Bitmap（JNI 环境）
jobject matToBitmap(JNIEnv* env, const cv::Mat& src);

// Bitmap 转 Mat（JNI 环境）
cv::Mat bitmapToMat(JNIEnv* env, jobject bitmap);

// =============================
// 绘制相关
// =============================

// 绘制人体关键点
void drawObjectKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color);

// 绘制人脸关键点
void drawObjectFaceKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color);

// 画检测框 + 文本 + 轨迹
void drawDetectionsOnFrame(cv::Mat& frame,
                           const std::vector<Object>& objects,
                           const char** class_names,
                           const unsigned char (*colors)[3],
                           int class_count);

// =============================
// 检测算法与推理流水线
// =============================

// 根据 modelId 创建对应的算法实例
IYoloAlgo* createModelInstance(int modelId);

// 统一推理流程：推理 -> 绘制 -> 统计摘要
std::vector<Object> detectAndUpdateSummary(cv::Mat& frame, double t0, double t1);

// 构造 DetectSummary 的 Java 对象
jobject createDetectSummaryJObject(JNIEnv* env, const char* className);

#endif // VISION_INFER_H


