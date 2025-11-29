#ifndef YOLO_COMMON_H
#define YOLO_COMMON_H
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <jni.h>

#include "globals.h"
#include "BYTETracker.h"
// Mat转Bitmap（JNI环境下）
jobject matToBitmap(JNIEnv* env, const cv::Mat& src);
// Bitmap转Mat（JNI环境下）
cv::Mat bitmapToMat(JNIEnv* env, jobject bitmap);
void drawObjectKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color);
void drawFaceLandmarks(cv::Mat& frame, const Object& obj, const unsigned char* color);
void drawObjectFaceKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color);
void drawFaceLandmarksByPart(cv::Mat& frame, const Object& obj, const unsigned char* color);
void analyzeFaceLandmarks(const Object& obj);
// 画检测框和类别
void drawDetectionsOnFrame(cv::Mat& frame, const std::vector<Object>& objects,const char** class_names, const unsigned char (*colors)[3],int class_count);
// 生成检测结果日志字符串
std::string buildDetectLog(const std::vector<Object>& objects, const char* class_names[]);
// 坐标还原到原图
void restoreObjectsToOriginal(std::vector<Object>& objects, int padX, int padY, float scale, int origW, int origH);
// 统计类别信息，返回日志字符串
std::string getClassCountLog(const std::vector<Object>& objects, const char* class_names[]);

#endif // YOLO_COMMON_H