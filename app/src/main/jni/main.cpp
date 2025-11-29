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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
extern "C" {
#include <libavformat/avformat.h>
}
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <platform.h>
#include <benchmark.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "globals.h"
#include "ndkcamera.h"
#include "IYoloAlgo.h"
#include "yolo_common.h"

#include "HighSpeed.h"
#include "YoloV8.h"
#include "Yolov8Seg.h"
#include "NanoDet.h"
#include "SimplePose.h"
#include "DbFace.h"
#include "FacelandMark.h"
#include "CombinedPoseFace.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static bool enableDetect = false;

// 相机帧渲染回调：MyNdkCamera类，继承自NdkCameraWindow，重载on_image_render用于推理和画框
class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

//实现：是否启用检测、执行YOLO推理、画框、类别统计、更新summary
void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    if (!enableDetect) {return;}
    std::vector<Object> objects;
    ncnn::MutexLockGuard g(g_lock);
    if (g_yolo) {
        double t1 = ncnn::get_current_time();
        g_yolo->detect(rgb, objects);
        // 后处理：画框、跟踪、拖尾等
        drawDetectionsOnFrame(rgb, objects,g_yolo->getClassNames(),g_yolo->getColors(), g_yolo->getClassCount());
        double t2 = ncnn::get_current_time();
        double allTime = (t2 - t1);
        // 直接使用推理时间的倒数计算FPS
        float fps = allTime > 0 ? 1000.0f / allTime : 0.0f;
        float cpu = get_cpu_usage();
        std::lock_guard<std::mutex> lock(g_summary_mutex);
        g_summary.allTimeMs = (float)allTime;
        g_summary.fps = fps;
        g_summary.cpuUsage = cpu;
        auto [logText, classInfo] = updateDetectSummary(objects, g_yolo->getClassNames());
        g_summary_cache = std::make_unique<DetectSummary>(DetectSummary{
                (float)g_summary.allTimeMs,
                (float)g_summary.inferTimeMs,
                (float)g_summary.fps,
                g_summary.cpuUsage,
                logText,
                classInfo});
    }
}
static MyNdkCamera* g_camera = 0;
extern "C" {
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    // 初始化FFmpeg网络组件（用于网络流支持）
    avformat_network_init();
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "FFmpeg network initialized");
    g_camera = new MyNdkCamera;
    g_camera = new MyNdkCamera;
    return JNI_VERSION_1_4;
}
JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(g_lock);
        delete g_yolo;
        g_yolo = 0;
    }
    delete g_camera;
    g_camera = 0;
    // 清理FFmpeg网络组件
    avformat_network_deinit();
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "FFmpeg network deinitialized");
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_LocalDetect_MainAlg_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu, jint inputsize)
{
    if (modelid < 0 || modelid > 10 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);
    // 动态选择算法
    if (g_yolo) { delete g_yolo; g_yolo = nullptr; }
    if (modelid == 0) {
        g_yolo = new HighSpeed();
    }else if (modelid == 1 || modelid == 2){
        g_yolo = new YoloV8();
    } else if (modelid == 3) {
        g_yolo = new Yolov8Seg();
    } else if (modelid == 4) {
        g_yolo = new NanoDet();
    } else if (modelid == 5) {
        g_yolo = new SimplePose();
    }else if (modelid == 6) {
        g_yolo = new DbFace();
    } else if (modelid == 7) {
        g_yolo = new FacelandMark();
    }  else if (modelid == 8) {
    g_yolo = new CombinedPoseFace();
    }
    bool use_gpu = (int)cpugpu == 1;
    if (g_yolo) {
        g_yolo->load(mgr, modelid, inputsize, use_gpu);
    }
    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_LocalDetect_MainAlg_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}
// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_LocalDetect_MainAlg_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_LocalDetect_MainAlg_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);
    g_camera->set_window(win);
    return JNI_TRUE;
}
// 新增 JNI 接口：设置置信度阈值
JNIEXPORT void JNICALL Java_com_tencent_LocalDetect_MainAlg_setThreshold(JNIEnv*, jobject, jfloat threshold)
{
    g_threshold = threshold;
}
// 新增 JNI 接口：设置nms阈值
JNIEXPORT void JNICALL Java_com_tencent_LocalDetect_MainAlg_setNms(JNIEnv*, jobject, jfloat nms)
{
    g_nms = nms;
}
JNIEXPORT void JNICALL
Java_com_tencent_LocalDetect_MainAlg_setTrackEnabled(JNIEnv* env, jobject thiz, jboolean enabled) {
    trackEnabled =enabled;
}
// JNI: 设置Shader拖尾开关
JNIEXPORT void JNICALL
Java_com_tencent_LocalDetect_MainAlg_setShaderEnabled(JNIEnv* env, jobject thiz, jboolean enabled) {
    shaderEnabled=enabled;
}
JNIEXPORT void JNICALL Java_com_tencent_LocalDetect_MainAlg_setDetectEnabled(JNIEnv*, jobject, jboolean enabled) {
    enableDetect = enabled;
}
JNIEXPORT jobject JNICALL Java_com_tencent_LocalDetect_MainAlg_getDetectSummary(JNIEnv *env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    if (!g_summary_cache) return nullptr;
    const DetectSummary& summary = *g_summary_cache;
    jclass cls = env->FindClass("com/tencent/LocalDetect/DetectSummary");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(FFFFLjava/lang/String;)V");
    jstring jlog = env->NewStringUTF(summary.logText.c_str());
    return env->NewObject(cls, ctor, summary.allTimeMs, summary.inferTimeMs, summary.fps, summary.cpuUsage, jlog);
}
JNIEXPORT jobject JNICALL Java_com_tencent_LocalDetect_MainAlg_detectImage(JNIEnv* env, jobject thiz, jobject bitmap, jint inputSize)
{
    double t0 = ncnn::get_current_time();
    AndroidBitmapInfo info;
    void* pixels;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return nullptr;
    cv::Mat orig_rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat orig_rgb;
    cv::cvtColor(orig_rgba, orig_rgb, cv::COLOR_RGBA2RGB);
    AndroidBitmap_unlockPixels(env, bitmap);
    // 1. 推理
    std::vector<Object> objects;
    if (g_yolo)
        g_yolo->detect(orig_rgb, objects);
    // 5. 在原图上画框、跟踪、拖尾
    drawDetectionsOnFrame(orig_rgb, objects,g_yolo->getClassNames(),g_yolo->getColors(), g_yolo->getClassCount());
    // 直接使用推理时间的倒数计算FPS
    double t1 = ncnn::get_current_time();
    double allTime = (t1 - t0);
    float fps = allTime > 0 ? 1000.0f / allTime : 0.0f;
    float cpu = get_cpu_usage();
    // 6 只在这里一次性写入g_summary
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.allTimeMs = (float)allTime;
    g_summary.fps = fps;
    g_summary.cpuUsage = cpu;
    auto [logText, classInfo] = updateDetectSummary(objects, g_yolo->getClassNames());
    g_summary_cache = std::make_unique<DetectSummary>(DetectSummary{
        (float)g_summary.allTimeMs,(float)g_summary.inferTimeMs, g_summary.fps, g_summary.cpuUsage, logText, classInfo});
    return matToBitmap(env, orig_rgb);}
}
