#include "vision_infer.h"

#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>

#include "IYoloAlgo.h"

// 算法头文件
#include "HighSpeed.h"
#include "YoloV8.h"
#include "Yolov8Seg.h"
#include "NanoDet.h"
#include "SimplePose.h"
#include "DbFace.h"
#include "FacelandMark.h"
#include "CombinedPoseFace.h"

// =============================
// Mat / Bitmap 互转
// =============================

jobject matToBitmap(JNIEnv* env, const cv::Mat& src)
{
    if (src.empty())
        return nullptr;

    cv::Mat rgba;
    if (src.channels() == 3)
    {
        cv::cvtColor(src, rgba, cv::COLOR_RGB2RGBA);
    }
    else if (src.channels() == 4)
    {
        rgba = src;
    }
    else
    {
        return nullptr;
    }

    jclass bitmapCls = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapFunc = env->GetStaticMethodID(
            bitmapCls, "createBitmap",
            "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");

    jstring configName = env->NewStringUTF("ARGB_8888");
    jclass bitmapConfigCls = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID valueOfFunc = env->GetStaticMethodID(
            bitmapConfigCls, "valueOf",
            "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;");

    jobject argbConfig = env->CallStaticObjectMethod(bitmapConfigCls, valueOfFunc, configName);
    jobject bitmap = env->CallStaticObjectMethod(
            bitmapCls, createBitmapFunc, rgba.cols, rgba.rows, argbConfig);

    void* pixels;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) == 0)
    {
        memcpy(pixels, rgba.data, rgba.total() * rgba.elemSize());
        AndroidBitmap_unlockPixels(env, bitmap);
    }

    env->DeleteLocalRef(configName);
    env->DeleteLocalRef(bitmapConfigCls);

    return bitmap;
}

cv::Mat bitmapToMat(JNIEnv* env, jobject bitmap)
{
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return cv::Mat();
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return cv::Mat();

    cv::Mat mat(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);

    AndroidBitmap_unlockPixels(env, bitmap);
    return bgr;
}

// =============================
// 绘制相关
// =============================

static BYTETracker g_tracker(25, 30);

void drawObjectKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color)
{
    if (obj.keyPoints.empty())
        return;

    float scale = std::max(frame.cols, frame.rows) / 640.0f;
    int point_thickness = std::max(3, int(3 * scale));
    int line_thickness  = std::max(2, int(2 * scale));

    // 人体关键点连接关系
    int joint_pairs[12][2] = {
            {0, 1}, {0, 2}, {2, 4}, {1, 3},
            {3, 5}, {0, 6}, {1, 7}, {6, 7},
            {6, 8}, {7, 9}, {8, 10}, {9, 11}
    };

    cv::Scalar point_color(0, 255, 0); // 绿色关键点
    cv::Scalar line_color(color[0], color[1], color[2]);

    // 绘制关键点
    for (const auto& kp : obj.keyPoints)
    {
        if (kp.prob > 0.1f)
        {
            cv::Point pt(static_cast<int>(kp.p.x), static_cast<int>(kp.p.y));
            cv::circle(frame, pt, point_thickness, point_color, -1);
        }
    }

    // 绘制骨骼连接线
    for (int i = 0; i < 12; i++)
    {
        int idx1 = joint_pairs[i][0];
        int idx2 = joint_pairs[i][1];

        if (idx1 < (int)obj.keyPoints.size() && idx2 < (int)obj.keyPoints.size())
        {
            const auto& kp1 = obj.keyPoints[idx1];
            const auto& kp2 = obj.keyPoints[idx2];

            if (kp1.prob > 0.1f && kp2.prob > 0.1f)
            {
                cv::Point pt1(static_cast<int>(kp1.p.x), static_cast<int>(kp1.p.y));
                cv::Point pt2(static_cast<int>(kp2.p.x), static_cast<int>(kp2.p.y));

                cv::Scalar current_line_color = line_color;
                if ((idx1 % 2 == 1) && (idx2 % 2 == 1) && idx1 >= 5 && idx2 >= 5)
                {
                    current_line_color = cv::Scalar(0, 0, 255); // 红色
                }

                cv::line(frame, pt1, pt2, current_line_color, line_thickness);
            }
        }
    }
}

void drawObjectFaceKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* /*color*/)
{
    float scale = std::max(frame.cols, frame.rows) / 640.0f;
    int point_thickness = std::max(2, int(2 * scale));

    for (size_t i = 0; i < obj.Face_keyPoints.size(); i++)
    {
        const auto& keypoint = obj.Face_keyPoints[i];
        cv::Point pt(static_cast<int>(keypoint.p.x), static_cast<int>(keypoint.p.y));

        int seed = static_cast<int>(i / 106) + 2020;
        std::srand(seed);
        int r = std::rand() % 256;
        int g = 125;
        int b = std::rand() % 256;

        cv::Scalar point_color(b, g, r);

        cv::circle(frame, pt, point_thickness, point_color, -1);
        cv::circle(frame, pt, point_thickness + 1, cv::Scalar(255, 255, 255), 1);
    }
}

void drawDetectionsOnFrame(cv::Mat& frame,
                           const std::vector<Object>& objects,
                           const char** class_names,
                           const unsigned char (*colors)[3],
                           int class_count)
{
    float scale = std::max(frame.cols, frame.rows) / 640.0f;
    int box_thickness  = std::max(2, int(2 * scale));
    double font_scale  = 0.5 * scale;
    int font_thickness = std::max(1, int(1 * scale));

    if (!trackEnabled)
    {
        for (const auto& obj : objects)
        {
            int cls = obj.label;
            if (cls < 0 || cls >= class_count)
                continue;

            float prob = obj.prob;
            const unsigned char* color = colors[cls % 10];
            cv::Scalar box_color(color[0], color[1], color[2]);
            cv::Rect rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(frame, rect, box_color, box_thickness);

            std::string info_text = cv::format("%s %.1f%%", class_names[cls], prob * 100);
            int baseline = 0;
            cv::Size info_size = cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX,
                                                 font_scale, font_thickness, &baseline);

            int text_x = std::max(0, std::min(rect.x, frame.cols - info_size.width));
            int text_y = std::max(0, rect.y - info_size.height - 3);

            cv::rectangle(frame,
                          cv::Rect(text_x - 2, text_y - 2,
                                   info_size.width + 4, info_size.height + 4),
                          box_color, -1);

            cv::putText(frame, info_text,
                        cv::Point(text_x, text_y + info_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale,
                        cv::Scalar(255, 255, 255), font_thickness);

            // 分割掩码
            if (!obj.markPoint.mask.empty())
            {
                for (int y = 0; y < frame.rows; y++)
                {
                    uchar* image_ptr = frame.ptr(y);
                    const float* mask_ptr = obj.markPoint.mask.ptr<float>(y);
                    for (int x = 0; x < frame.cols; x++)
                    {
                        if (mask_ptr[x] >= 0.5f)
                        {
                            image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                            image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                            image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
                        }
                        image_ptr += 3;
                    }
                }
            }

            if (!obj.Face_keyPoints.empty())
            {
                drawObjectFaceKeypoints(frame, obj, color);
            }
            if (!obj.keyPoints.empty())
            {
                drawObjectKeypoints(frame, obj, color);
            }
        }

        return;
    }

    std::vector<STrack> output_stracks = g_tracker.update(objects);
    for (size_t i = 0; i < output_stracks.size(); i++)
    {
        const auto& track = output_stracks[i];
        std::vector<float> tlwh = track.tlwh;
        float prob  = track.score;
        int cls     = track.cls;
        int track_id = track.track_id;

        // 拖尾（轨迹）
        if (shaderEnabled)
        {
            const auto& history = track.trajectory;
            for (size_t j = 1; j < history.size(); j++)
            {
                double alpha = 0.3f + 0.7f * (j + 1) / history.size();
                const unsigned char* color = colors[cls % 10];
                cv::Scalar line_color(color[0] * alpha, color[1] * alpha, color[2] * alpha);
                cv::line(frame, history[j - 1], history[j], line_color, font_thickness);
            }
        }

        const unsigned char* color = colors[cls % 19];
        cv::Scalar track_color(color[0], color[1], color[2]);
        cv::Rect rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);

        cv::rectangle(frame, rect, track_color, box_thickness);

        int baseline = 0;

        // ID（框内左上角）
        std::string id_text = std::to_string(track_id);
        cv::Size id_size = cv::getTextSize(id_text, cv::FONT_HERSHEY_SIMPLEX,
                                           font_scale, font_thickness, &baseline);
        cv::Point id_pos(rect.x + 3, rect.y + id_size.height + 3);
        cv::putText(frame, id_text, id_pos, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(255, 0, 0), box_thickness);

        // 类别 + 置信度（框外上方）
        std::string info_text = cv::format("%s %.1f%%", class_names[cls], prob * 100);
        cv::Size info_size = cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX,
                                             font_scale, font_thickness, &baseline);

        int text_x = std::max(0, std::min(rect.x, frame.cols - info_size.width));
        int text_y = std::max(0, rect.y - info_size.height - 3);

        cv::rectangle(frame,
                      cv::Rect(text_x - 2, text_y - 2,
                               info_size.width + 4, info_size.height + 4),
                      track_color, -1);
        cv::putText(frame, info_text,
                    cv::Point(text_x, text_y + info_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(255, 255, 255), font_thickness);
    }
}

// =============================
// 模型工厂 & 推理流水线
// =============================

IYoloAlgo* createModelInstance(int modelId)
{
    IYoloAlgo* model = nullptr;

    if (modelId == 0)
    {
        model = new HighSpeed();
    }
    else if (modelId == 1 || modelId == 2)
    {
        model = new YoloV8();
    }
    else if (modelId == 3)
    {
        model = new Yolov8Seg();
    }
    else if (modelId == 4)
    {
        model = new NanoDet();
    }
    else if (modelId == 5)
    {
        model = new SimplePose();
    }
    else if (modelId == 6)
    {
        model = new DbFace();
    }
    else if (modelId == 7)
    {
        model = new FacelandMark();
    }
    else if (modelId == 8)
    {
        model = new CombinedPoseFace();
    }

    return model;
}

std::vector<Object> detectAndUpdateSummary(cv::Mat& frame, double t0, double /*t1*/)
{
    std::vector<Object> objects;

    // 推理
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_yolo)
        {
            g_yolo->detect(frame, objects);
        }
        else
        {
            return objects;
        }
    }

    // 绘制（框 / 分割 / 关键点 / 轨迹）
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_yolo)
        {
            drawDetectionsOnFrame(frame, objects,
                                  g_yolo->getClassNames(),
                                  g_yolo->getColors(),
                                  g_yolo->getClassCount());
        }
    }

    // 计算 FPS & 更新摘要
    double t2 = ncnn::get_current_time();
    double allTime = (t2 - t0);
    float fps = allTime > 0 ? 1000.0f / (float)allTime : 0.0f;

    std::lock_guard<std::mutex> lock(g_summary_mutex);
    g_summary.allTimeMs = (float)allTime;
    // inferTimeMs 在当前实现中暂未单独统计
    float inferTimeMs = g_summary.inferTimeMs;
    g_summary.fps = fps;

    // 更新类别统计文本
    std::string logText;
    std::vector<std::string> classInfo;
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_yolo)
        {
            auto result = updateDetectSummary(objects, g_yolo->getClassNames());
            logText = result.first;
            classInfo = result.second;
        }
    }

    g_summary_cache = std::make_unique<DetectSummary>(DetectSummary{
            g_summary.allTimeMs,
            inferTimeMs,
            g_summary.fps,
            logText,
            classInfo
    });

    return objects;
}

jobject createDetectSummaryJObject(JNIEnv* env, const char* className)
{
    std::lock_guard<std::mutex> lock(g_summary_mutex);
    if (!g_summary_cache)
        return nullptr;

    const DetectSummary& summary = *g_summary_cache;

    jclass cls = env->FindClass(className);
    if (cls == nullptr)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn",
                            "Failed to find DetectSummary class: %s", className);
        env->ExceptionClear();
        return nullptr;
    }

    jmethodID ctor = env->GetMethodID(cls, "<init>", "(FFFLjava/lang/String;)V");
    if (ctor == nullptr)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn",
                            "Failed to find DetectSummary constructor");
        env->ExceptionClear();
        env->DeleteLocalRef(cls);
        return nullptr;
    }

    jstring jlog = env->NewStringUTF(summary.logText.c_str());
    if (jlog == nullptr)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn",
                            "Failed to create jstring");
        env->ExceptionClear();
        env->DeleteLocalRef(cls);
        return nullptr;
    }

    jobject result = env->NewObject(cls, ctor,
                                    summary.allTimeMs,
                                    summary.inferTimeMs,
                                    summary.fps,
                                    jlog);

    env->DeleteLocalRef(jlog);
    env->DeleteLocalRef(cls);

    if (result == nullptr)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn",
                            "Failed to create DetectSummary object");
        env->ExceptionClear();
    }

    return result;
}


