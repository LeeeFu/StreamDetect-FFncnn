#include "yolo_common.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <android/bitmap.h>
#include <algorithm>
// Mat转Bitmap（JNI环境下）ffmpeg
jobject matToBitmap(JNIEnv* env, const cv::Mat& src) {
    if (src.empty()) return nullptr;
    cv::Mat rgba;
    if (src.channels() == 3) {
        cv::cvtColor(src, rgba, cv::COLOR_RGB2RGBA);
    } else if (src.channels() == 4) {
        rgba = src;
    } else {
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
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) == 0) {
        memcpy(pixels, rgba.data, rgba.total() * rgba.elemSize());
        AndroidBitmap_unlockPixels(env, bitmap);
    }
    env->DeleteLocalRef(configName);
    env->DeleteLocalRef(bitmapConfigCls);
    return bitmap;
}
// Bitmap转Mat（JNI环境下）
cv::Mat bitmapToMat(JNIEnv* env, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return cv::Mat();
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return cv::Mat();
    cv::Mat mat(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);
    return bgr;
}
BYTETracker tracker(25, 30);
// 绘制单个对象的关键点
void drawObjectKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color) {
    if (obj.keyPoints.empty()) return;
    float scale = std::max(frame.cols, frame.rows) / 640.0f;
    int point_thickness = std::max(3, int(3 * scale));
    int line_thickness = std::max(2, int(2 * scale));
    // 人体关键点连接关系（17个关键点）
    // 0 nose, 1 left_eye, 2 right_eye, 3 left_Ear, 4 right_Ear, 5 left_Shoulder, 6 rigth_Shoulder,
    // 7 left_Elbow, 8 right_Elbow, 9 left_Wrist, 10 right_Wrist, 11 left_Hip, 12 right_Hip,
    // 13 left_Knee, 14 right_Knee, 15 left_Ankle, 16 right_Ankle
//    int joint_pairs[16][2] = {
//            {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8},
//            {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
//    };
    int joint_pairs[12][2] = {
             {0, 1}, {0, 2}, {2, 4}, {1, 3},
            {3, 5}, {0, 6}, {1, 7}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}
    };
    cv::Scalar point_color(0, 255, 0); // 绿色关键点
    cv::Scalar line_color(color[0], color[1], color[2]); // 使用对象颜色
    // 绘制关键点
    for (const auto& kp : obj.keyPoints) {
        if (kp.prob > 0.1f) { // 只绘制置信度较高的关键点
            cv::Point pt(static_cast<int>(kp.p.x), static_cast<int>(kp.p.y));
            cv::circle(frame, pt, point_thickness, point_color, -1);
        }
    }
    // 绘制骨骼连接线
    for (int i = 0; i < 12; i++) {
        int idx1 = joint_pairs[i][0];
        int idx2 = joint_pairs[i][1];

        if (idx1 < obj.keyPoints.size() && idx2 < obj.keyPoints.size()) {
            const auto& kp1 = obj.keyPoints[idx1];
            const auto& kp2 = obj.keyPoints[idx2];
            // 只绘制两个关键点都有足够置信度的连接
            if (kp1.prob > 0.1f && kp2.prob > 0.1f) {
                cv::Point pt1(static_cast<int>(kp1.p.x), static_cast<int>(kp1.p.y));
                cv::Point pt2(static_cast<int>(kp2.p.x), static_cast<int>(kp2.p.y));

                // 人体左侧用红色，右侧用对象颜色
                cv::Scalar current_line_color = line_color;
                if ((idx1 % 2 == 1) && (idx2 % 2 == 1) && idx1 >= 5 && idx2 >= 5) {
                    current_line_color = cv::Scalar(0, 0, 255); // 红色
                }
                cv::line(frame, pt1, pt2, current_line_color, line_thickness);
            }
        }
    }
}
void drawObjectFaceKeypoints(cv::Mat& frame, const Object& obj, const unsigned char* color){
    float scale = std::max(frame.cols, frame.rows) / 640.0f;
    int point_thickness = std::max(2, int(2 * scale));
    // 参考代码中的随机颜色生成逻辑
    for (size_t i = 0; i < obj.Face_keyPoints.size(); i++) {
        const auto& keypoint = obj.Face_keyPoints[i];
        cv::Point pt(static_cast<int>(keypoint.p.x), static_cast<int>(keypoint.p.y));

        // 使用固定的种子确保相同位置的关键点颜色一致
        int seed = i / 106 + 2020;
        std::srand(seed);
        int r = std::rand() % 256;
        int g = 125; // 固定绿色值
        int b = std::rand() % 256;

        cv::Scalar point_color(b, g, r); // OpenCV使用BGR格式
        // 绘制关键点，使用参考代码中的样式
        cv::circle(frame, pt, point_thickness, point_color, -1);
        // 添加白色边框增强可见性
        cv::circle(frame, pt, point_thickness + 1, cv::Scalar(255, 255, 255), 1);
    }
}
// 画检测框和类别
void drawDetectionsOnFrame(cv::Mat& frame, const std::vector<Object>& objects,const char** class_names,const unsigned char (*colors)[3], int class_count) {
    float scale = std::max(frame.cols, frame.rows) / 640.0f; // 以640为基准
    int box_thickness = std::max(2, int(2 * scale));
    double font_scale = 0.5 * scale;
    int font_thickness = std::max(1, int(1 * scale));
    if (!trackEnabled) {
        for (const auto& obj : objects) {
            int cls = obj.label;
            if (cls < 0 || cls >= class_count) continue; // 防止越界
            float prob = obj.prob;
            const unsigned char* color = colors[cls % 10];
            cv::Scalar box_color(color[0], color[1], color[2]);
            cv::Rect rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
            cv::rectangle(frame, rect, box_color, box_thickness);
            std::string info_text = cv::format("%s %.1f%%", class_names[cls], prob * 100);
            int baseline = 0;
            cv::Size info_size = cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
            int text_x = std::max(0, std::min(rect.x, frame.cols - info_size.width));
            int text_y = std::max(0, rect.y - info_size.height - 3);
            cv::rectangle(frame, cv::Rect(text_x - 2, text_y - 2, info_size.width + 4, info_size.height + 4), box_color, -1);
            cv::putText(frame, info_text, cv::Point(text_x, text_y + info_size.height), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);
            // 1. 分割掩码绘制（如果有mask）
            if (!obj.markPoint.mask.empty())
            {
                for (int y = 0; y < frame.rows; y++) {
                    uchar* image_ptr = frame.ptr(y);
                    const float* mask_ptr = obj.markPoint.mask.ptr<float>(y);
                    for (int x = 0; x < frame.cols; x++) {
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
            if (!obj.Face_keyPoints.empty()) {
                // 人脸关键点绘制
                drawObjectFaceKeypoints(frame, obj, color);
            }
            if (!obj.keyPoints.empty()) {
                // 人体关键点绘制
                drawObjectKeypoints(frame, obj, color);
            }
        }
        return;
    }
    vector<STrack> output_stracks = tracker.update(objects);
    for (int i = 0; i < output_stracks.size(); i++)
    {
        const auto& track = output_stracks[i];
        vector<float> tlwh = track.tlwh;
        float prob =track.score;
        int cls =track.cls;
        int track_id = track.track_id;
        // 拖尾（轨迹）功能，只有shaderEnabled为true时才画
        if (shaderEnabled) {
            const auto& history = track.trajectory;
            for (size_t j = 1; j < history.size(); j++) {
                double alpha = 0.3f + 0.7f * (j + 1) / history.size();
                const unsigned char* color = colors[cls % 10];
                //画点
                //       cv::Scalar point_color(color[0] * alpha, color[1] * alpha, color[2] * alpha);
                //       cv::circle(rgb, history[j], 2, point_color, -1);
                //画线
                cv::Scalar line_color(color[0] * alpha, color[1] * alpha, color[2] * alpha);
                cv::line(frame, history[j-1], history[j], line_color, font_thickness);
            }
        }
        const unsigned char* color = colors[cls%19];
        cv::Scalar track_color(color[0], color[1], color[2]);
        cv::Rect rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
        // 绘制检测框
        cv::rectangle(frame, rect, track_color, box_thickness);
        int baseline = 0;
        //1. 绘制ID（框内左上角，无背景）
        std::string id_text = std::to_string(track.track_id);
        cv::Size id_size = cv::getTextSize(id_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        // ID位置：框内左上角，偏移3像素
        cv::Point id_pos(rect.x + 3, rect.y + id_size.height + 3);
        cv::putText(frame, id_text, id_pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 0, 0), box_thickness);
        std::string info_text = cv::format("%s %.1f%%", class_names[cls], prob * 100);
        cv::Size info_size = cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        // 计算文本背景位置（框外正上方）
        int text_x = std::max(0, std::min(rect.x, frame.cols - info_size.width));
        int text_y = std::max(0, rect.y - info_size.height - 3);
        cv::rectangle(frame, cv::Rect(text_x - 2, text_y - 2, info_size.width + 4, info_size.height + 4), track_color, -1);
        cv::putText(frame, info_text, cv::Point(text_x, text_y + info_size.height), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);
    }
}
// 生成检测结果日志字符串
std::string buildDetectLog(const std::vector<Object>& objects, const char* class_names[]) {
    std::map<int, int> cls_count;
    for (const auto& obj : objects) cls_count[obj.label]++;
    std::string logLine;
    for (const auto& kv : cls_count) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s: %d", class_names[kv.first], kv.second);
        if (!logLine.empty()) logLine += ", ";
        logLine += buf;
    }
    return logLine.empty() ? "None" : logLine;
}
// 坐标还原到原图
void restoreObjectsToOriginal(std::vector<Object>& objects, int padX, int padY, float scale, int origW, int origH) {
    for (auto& obj : objects) {
        obj.rect.x = (obj.rect.x - padX) / scale;
        obj.rect.y = (obj.rect.y - padY) / scale;
        obj.rect.width /= scale;
        obj.rect.height /= scale;
        obj.rect.x = std::max(0.f, std::min(obj.rect.x, (float)(origW - 1)));
        obj.rect.y = std::max(0.f, std::min(obj.rect.y, (float)(origH - 1)));
        obj.rect.width = std::max(0.f, std::min(obj.rect.width, (float)(origW - obj.rect.x)));
        obj.rect.height = std::max(0.f, std::min(obj.rect.height, (float)(origH - obj.rect.y)));
        // 还原人体关键点坐标
        for (auto& kp : obj.keyPoints) {
            kp.p.x = (kp.p.x - padX) / scale;
            kp.p.y = (kp.p.y - padY) / scale;
            // 边界检查
            kp.p.x = std::max(0.f, std::min(kp.p.x, (float)(origW - 1)));
            kp.p.y = std::max(0.f, std::min(kp.p.y, (float)(origH - 1)));
        }
        // 还原人脸关键点坐标
        for (auto& kp : obj.Face_keyPoints) {
            kp.p.x = (kp.p.x - padX) / scale;
            kp.p.y = (kp.p.y - padY) / scale;
            // 边界检查
            kp.p.x = std::max(0.f, std::min(kp.p.x, (float)(origW - 1)));
            kp.p.y = std::max(0.f, std::min(kp.p.y, (float)(origH - 1)));
        }
    }
}
// 统计类别信息，返回日志字符串
std::string getClassCountLog(const std::vector<Object>& objects, const char* class_names[]) {
    std::map<int, int> cls_count;
    for (const auto& obj : objects) cls_count[obj.label]++;
    std::string logLine;
    for (const auto& kv : cls_count) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s: %d", class_names[kv.first], kv.second);
        if (!logLine.empty()) logLine += ", ";
        logLine += buf;
    }
    return logLine.empty() ? "None" : logLine;
}
