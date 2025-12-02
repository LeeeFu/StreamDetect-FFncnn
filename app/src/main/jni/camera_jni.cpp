// 相机与单张图片检测 JNI 入口
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

#include "vision_base.h"
#include "vision_infer.h"
#include "ndkcamera.h"
#include "IYoloAlgo.h"
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

// 实现：是否启用检测、执行YOLO推理、画框、类别统计、更新summary
void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    if (!enableDetect)
        return;

    double t0 = ncnn::get_current_time();
    double t1 = t0;
    // 使用公共函数执行推理和更新摘要
    detectAndUpdateSummary(rgb, t0, t1);
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
JNIEXPORT jboolean JNICALL
Java_NcnnTencent_common_JniBridge_loadModel(JNIEnv* env, jobject thiz,
                                                jobject assetManager,
                                                jint modelid, jint cpugpu,
                                                jint inputsize)
{
    if (modelid < 0 || modelid > 10 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    // 使用公共函数创建模型
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_yolo)
        {
            delete g_yolo;
            g_yolo = nullptr;
        }
        g_yolo = createModelInstance(modelid);
    }

    bool use_gpu = (int)cpugpu == 1;
    if (g_yolo)
    {
        g_yolo->load(mgr, modelid, inputsize, use_gpu);
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL
Java_NcnnTencent_common_JniBridge_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL
Java_NcnnTencent_common_JniBridge_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL
Java_NcnnTencent_common_JniBridge_setOutputWindow(JNIEnv* env, jobject thiz,
                                                      jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);
    g_camera->set_window(win);
    return JNI_TRUE;
}

// JNI 接口：设置置信度阈值
JNIEXPORT void JNICALL
Java_NcnnTencent_common_JniBridge_setThreshold(JNIEnv*, jobject, jfloat threshold)
{
    g_threshold = threshold;
}

// JNI 接口：设置 NMS 阈值
JNIEXPORT void JNICALL
Java_NcnnTencent_common_JniBridge_setNms(JNIEnv*, jobject, jfloat nms)
{
    g_nms = nms;
}

JNIEXPORT void JNICALL
Java_NcnnTencent_common_JniBridge_setTrackEnabled(JNIEnv* env, jobject thiz,
                                                      jboolean enabled)
{
    trackEnabled = enabled;
}

// JNI 接口：设置 Shader 拖尾开关
JNIEXPORT void JNICALL
Java_NcnnTencent_common_JniBridge_setShaderEnabled(JNIEnv* env, jobject thiz,
                                                       jboolean enabled)
{
    shaderEnabled = enabled;
}

JNIEXPORT void JNICALL
Java_NcnnTencent_common_JniBridge_setDetectEnabled(JNIEnv*, jobject,
                                                       jboolean enabled)
{
    enableDetect = enabled;
}

JNIEXPORT jobject JNICALL
Java_NcnnTencent_common_JniBridge_getDetectSummary(JNIEnv *env, jobject thiz)
{
    return createDetectSummaryJObject(env,
                                      "NcnnTencent/common/Models$DetectSummary");
}

JNIEXPORT jobject JNICALL
Java_NcnnTencent_common_JniBridge_detectImage(JNIEnv* env, jobject thiz,
                                                  jobject bitmap, jint inputSize)
{
    double t0 = ncnn::get_current_time();

    AndroidBitmapInfo info;
    void* pixels;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return nullptr;

    cv::Mat orig_rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat orig_rgb;
    cv::cvtColor(orig_rgba, orig_rgb, cv::COLOR_RGBA2RGB);
    AndroidBitmap_unlockPixels(env, bitmap);

    // 使用公共函数执行推理和更新摘要
    double t1 = ncnn::get_current_time();
    detectAndUpdateSummary(orig_rgb, t0, t1);

    return matToBitmap(env, orig_rgb);
}

} // extern "C"
