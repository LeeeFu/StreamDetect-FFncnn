#include <jni.h>
#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
// OpenCV headers - 明确包含需要的模块
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// FFmpeg headers - 使用extern "C"包装
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/error.h>
#include <libavutil/display.h>
#include <libavutil/dict.h>
}
#include <mutex>

#include "globals.h"
#include "IYoloAlgo.h"
#include "yolo_common.h"

#include "HighSpeed.h"
#include "YoloV8.h"
#include "Yolov8Seg.h"
#include "NanoDet.h"
#include "SimplePose.h"

// 视频解码器类
class FFmpegVideoDecoder {
private:
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    SwsContext* sws_ctx;
    AVFrame* frame;
    AVFrame* rgb_frame;
    AVPacket* packet;
    uint8_t* rgb_buffer;
    int video_stream_index;
    int width, height;
    std::atomic<bool> stop_flag;
    int rotation; // 新增，记录视频旋转角度
    int last_error_code; // 保存最后的错误码
    std::string last_error_msg; // 保存最后的错误信息

public:
    AVFormatContext* getFormatCtx() const { return format_ctx; }
    int getVideoStreamIndex() const { return video_stream_index; }

    FFmpegVideoDecoder() : format_ctx(nullptr), codec_ctx(nullptr),
                           sws_ctx(nullptr), frame(nullptr), rgb_frame(nullptr),
                           packet(nullptr), rgb_buffer(nullptr),
                           video_stream_index(-1), width(0), height(0), stop_flag(false), rotation(0),
                           last_error_code(0), last_error_msg("") {}

    // 获取最后的错误信息
    std::string getLastError() const {
        if (last_error_code != 0) {
            return "FFmpeg错误码: " + std::to_string(last_error_code) + ", " + last_error_msg;
        }
        return last_error_msg;
    }

    ~FFmpegVideoDecoder() {
        cleanup();
    }

    bool init(const char* filename) {
        // 初始化FFmpeg
        // 打开输入文件
//        if (avformat_open_input(&format_ctx, filename, nullptr, nullptr) < 0) {
//            return false;
//        }
// 设置网络流选项
        AVDictionary* options = nullptr;
        // 判断是否为网络流
        bool isNetworkStream = (strstr(filename, "http://") != nullptr ||
                                strstr(filename, "https://") != nullptr ||
                                strstr(filename, "rtsp://") != nullptr ||
                                strstr(filename, "rtmp://") != nullptr);

        if (isNetworkStream) {
            // 设置超时时间（10秒）
            av_dict_set(&options, "timeout", "10000000", 0); // 微秒，10秒
            // 设置TCP超时
            av_dict_set(&options, "tcp_timeout", "10000000", 0);
            // 设置User-Agent（某些服务器需要）
            av_dict_set(&options, "user_agent", "FFmpeg/Android", 0);
            // 对于RTSP，使用TCP传输（更稳定）
            if (strstr(filename, "rtsp://") != nullptr) {
                av_dict_set(&options, "rtsp_transport", "tcp", 0);
            }
            // 对于HTTP，允许重定向
            if (strstr(filename, "http://") != nullptr || strstr(filename, "https://") != nullptr) {
                av_dict_set(&options, "follow_redirect", "1", 0);
                av_dict_set(&options, "multiple_requests", "1", 0);
            }
            // 设置缓冲区大小
            av_dict_set(&options, "buffer_size", "10485760", 0); // 10MB
            // 允许不安全的SSL连接（如果需要）
            av_dict_set(&options, "verify_ssl", "0", 0);

            __android_log_print(ANDROID_LOG_INFO, "FFmpegVideoDetect", "Opening network stream: %s", filename);
        }

        // 打开输入文件/流
        int ret = avformat_open_input(&format_ctx, filename, nullptr, &options);
        if (options) {
            av_dict_free(&options);
        }
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            __android_log_print(ANDROID_LOG_ERROR, "FFmpegVideoDetect", "Failed to open input: %s, error: %s", filename, errbuf);
            return false;
        }
        ///

        // 查找流信息
//        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
//            return false;
//        }

        // 查找流信息（对于网络流，需要更长的超时时间）
        AVDictionary* stream_options = nullptr;
        if (isNetworkStream) {
            // 设置查找流信息的超时时间（15秒）
            av_dict_set(&stream_options, "timeout", "15000000", 0);
        }

        ret = avformat_find_stream_info(format_ctx, &stream_options);
        if (stream_options) {
            av_dict_free(&stream_options);
        }
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            __android_log_print(ANDROID_LOG_ERROR, "FFmpegVideoDetect", "Failed to find stream info: error code: %d, error: %s", ret, errbuf);
            last_error_code = ret;
            last_error_msg = std::string("查找流信息失败: ") + errbuf;
            return false;
        }
        //////

        // 查找视频流
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                break;
            }
        }

        if (video_stream_index == -1) {
            __android_log_print(ANDROID_LOG_ERROR, "FFmpegVideoDetect", "No video stream found in file: %s", filename);
            return false; // 无视频流
        }

        // 获取解码器
        AVCodecParameters* codecpar = format_ctx->streams[video_stream_index]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            return false;
        }

        // 创建解码器上下文
        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            return false;
        }

        if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
            return false;
        }

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            return false;
        }

        width = codec_ctx->width;
        height = codec_ctx->height;

        if (width == 0 || height == 0) {
            __android_log_print(ANDROID_LOG_ERROR, "FFmpegVideoDetect", "Invalid video size: %dx%d", width, height);
            return false; // 无效分辨率
        }

        // 分配帧缓冲区
        frame = av_frame_alloc();
        rgb_frame = av_frame_alloc();
        packet = av_packet_alloc();

        if (!frame || !rgb_frame || !packet) {
            return false;
        }

        // 分配RGB缓冲区
        int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
        rgb_buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
        if (!rgb_buffer) {
            return false;
        }

        av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buffer,
                             AV_PIX_FMT_RGB24, width, height, 1);

        // 创建图像转换上下文
        sws_ctx = sws_getContext(width, height, codec_ctx->pix_fmt,
                                 width, height, AV_PIX_FMT_RGB24,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        // 新增：读取旋转元数据
        rotation = 0;
        uint8_t* display_matrix = av_stream_get_side_data(
                format_ctx->streams[video_stream_index], AV_PKT_DATA_DISPLAYMATRIX, NULL);
        if (display_matrix) {
            double rot = av_display_rotation_get((int32_t*)display_matrix);
            rotation = (int)round(rot);
            // 只允许0/90/180/270
            if (rotation % 90 != 0) rotation = 0;
            if (rotation < 0) rotation += 360;
        }
        return true;
    }

    void stop() {
        stop_flag = true;
    }

    bool end_of_stream = false; // 新增，标记是否已读完所有包
    bool decode_frame(cv::Mat& output_frame) {
        if (stop_flag) return false;
        while (true) {
            if (!end_of_stream) {
                int ret = av_read_frame(format_ctx, packet);
                if (ret < 0) {
                    end_of_stream = true;
                    av_packet_unref(packet);
                    __android_log_print(ANDROID_LOG_INFO, "FFmpegVideoDetect", "Enter flush mode");
                } else if (packet->stream_index == video_stream_index) {
                    int response = avcodec_send_packet(codec_ctx, packet);
                    __android_log_print(ANDROID_LOG_INFO, "FFmpegVideoDetect", "Send packet, response=%d", response);
                    av_packet_unref(packet);
                    if (response < 0) continue;
                } else {
                    av_packet_unref(packet);
                    continue;
                }
            }
            int response = avcodec_receive_frame(codec_ctx, frame);
            __android_log_print(ANDROID_LOG_INFO, "FFmpegVideoDetect", "Receive frame, response=%d", response);
            if (response == AVERROR(EAGAIN)) {
                if (end_of_stream) return false;
                continue;
            } else if (response == AVERROR_EOF) {
                return false;
            } else if (response < 0) {
                return false;
            }
            __android_log_print(ANDROID_LOG_INFO, "FFmpegVideoDetect", "Decoded frame: %d x %d, pix_fmt=%d", frame->width, frame->height, frame->format);
            sws_scale(sws_ctx, frame->data, frame->linesize, 0, height, rgb_frame->data, rgb_frame->linesize);
            output_frame = cv::Mat(height, width, CV_8UC3, rgb_buffer, rgb_frame->linesize[0]);
            output_frame = output_frame.clone();
            // 新增：根据rotation旋转帧
            if (rotation == 90) {
                cv::rotate(output_frame, output_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            } else if (rotation == 180) {
                    cv::rotate(output_frame, output_frame, cv::ROTATE_180);
            } else if (rotation == 270) {
                cv::rotate(output_frame, output_frame, cv::ROTATE_90_CLOCKWISE);
            }
            return true;
        }
    }

    void cleanup() {
        stop_flag = true;

        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }

        if (rgb_buffer) {
            av_free(rgb_buffer);
            rgb_buffer = nullptr;
        }

        if (packet) {
            av_packet_free(&packet);
        }

        if (frame) {
            av_frame_free(&frame);
        }

        if (rgb_frame) {
            av_frame_free(&rgb_frame);
        }

        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }

        if (format_ctx) {
            avformat_close_input(&format_ctx);
        }
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
};
extern "C"
JNIEXPORT void JNICALL
Java_com_tencent_LocalDetect_MainActivity_startFFmpegVideoDetect(
        JNIEnv* env, jobject thiz, jstring videoPath, jint inputSize,
        jobject callback, jobject assetManager, jint modelid, jint cpugpu) {

    const char* path = env->GetStringUTFChars(videoPath, 0);

    // 获取回调方法ID - 修改为只传递Bitmap，不再传递RectF数组
    jclass cbCls = env->GetObjectClass(callback);
    jmethodID onFrame = env->GetMethodID(cbCls, "onFrame", "(Landroid/graphics/Bitmap;[Landroid/graphics/RectF;)V");
    jmethodID onFinish = env->GetMethodID(cbCls, "onFinish", "()V");
    jmethodID onError = env->GetMethodID(cbCls, "onError", "(Ljava/lang/String;)V"); // 新增

    // 检查模型是否已加载
    {
        ncnn::MutexLockGuard g(g_lock);
        if (!g_yolo) {
            jstring msg = env->NewStringUTF("模型未加载");
            env->CallVoidMethod(callback, onError, msg);
            env->DeleteLocalRef(msg);
            env->ReleaseStringUTFChars(videoPath, path);
            return;
        }
    }
    // 创建FFmpeg解码器
    FFmpegVideoDecoder decoder;
    if (!decoder.init(path)) {
        jstring msg = env->NewStringUTF("视频无视频流或解码器初始化失败");
        env->CallVoidMethod(callback, onError, msg);
        env->DeleteLocalRef(msg);
        env->ReleaseStringUTFChars(videoPath, path);
        return;
    }
    int frameCount = 0;
    int processedFrames = 0;
    const int maxFrames = 1000;
    bool hasFrameCallback = false;   // 新增，标记是否有帧回调
    cv::Mat rgb_frame;
    while (frameCount < maxFrames) {
        double t0 = ncnn::get_current_time();
        bool decodeSuccess = decoder.decode_frame(rgb_frame);
        if (!decodeSuccess) break;
        frameCount++;
        // 固定间隔抽帧：每隔一帧处理一次（对于25FPS视频，处理速度约12.5FPS）
        if (frameCount % 2 == 0) {
            // 跳过这一帧，只解码不处理
            continue;
        }
        // 处理这一帧（推理+回调）
        double t1 = ncnn::get_current_time();
        // 3. 推理
        std::vector<Object> objects;
        ncnn::MutexLockGuard g(g_lock);
        if (g_yolo)
            g_yolo->detect(rgb_frame, objects);
        double t2 = ncnn::get_current_time();
        double ffmpeg_allTime = (t2 - t1) + 0.5 * (t1 - t0);
        // 5. 在JNI层直接画框到原帧上（优化：避免Java层绘制）
        drawDetectionsOnFrame(rgb_frame, objects, g_yolo->getClassNames(),g_yolo->getColors(), g_yolo->getClassCount());
        // 直接使用推理时间的倒数计算FPS
        double fps = ffmpeg_allTime > 0 ? 1000.0f / ffmpeg_allTime : 0.0f;
        // 获取CPU使用率（与摄像头检测保持一致）
        float cpu = get_cpu_usage();
        g_summary.allTimeMs = (float)ffmpeg_allTime;
        g_summary.fps = fps;
        g_summary.cpuUsage = cpu;
        auto [logText, classInfo] = updateDetectSummary(objects, g_yolo->getClassNames());
        g_summary_cache = std::make_unique<DetectSummary>(DetectSummary{
                (float)g_summary.allTimeMs,
                (float)g_summary.inferTimeMs, // 若有单独推理耗时可替换
                (float)g_summary.fps,
                g_summary.cpuUsage,
                logText,
                classInfo});
        // 6. Bitmap转换
        jclass rectFCls = env->FindClass("android/graphics/RectF");
        jobjectArray rectFArray = env->NewObjectArray(0, rectFCls, nullptr);
        jobject bitmap = matToBitmap(env, rgb_frame);
        if (!bitmap) {
            env->DeleteLocalRef(rectFArray);
            continue;
        }
        // 8. 回调onFrame（固定间隔抽帧后回调）
        processedFrames++;
        hasFrameCallback = true; // 有帧回调
        env->CallVoidMethod(callback, onFrame, bitmap, rectFArray);
        // 9. 释放局部引用
        env->DeleteLocalRef(bitmap);
        env->DeleteLocalRef(rectFArray);
    }
    // 清理资源
    decoder.cleanup();
    if (!hasFrameCallback) {
        jstring msg = env->NewStringUTF("视频无帧或内容损坏");
        env->CallVoidMethod(callback, onError, msg);
        env->DeleteLocalRef(msg);
    } else {
        env->CallVoidMethod(callback, onFinish);
    }
    env->ReleaseStringUTFChars(videoPath, path);
}

// =========================
// 云端 NetworkVideoManager 接口
// 使用同一套 FFmpeg 解码 + YOLO 推理，实现网络流预览
// =========================

static std::thread g_network_thread;
static std::atomic<bool> g_network_running(false);
static std::atomic<bool> g_network_stop(false);

extern "C"
JNIEXPORT void JNICALL
Java_com_tencent_CloudDetect_NetworkVideoManager_startNetworkVideoStream(
        JNIEnv* env, jobject thiz, jstring jurl, jint inputSize,
        jint modelid, jint cpugpu) {

    if (g_network_running.load()) {
        __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "network stream already running");
        return;
    }

    const char* url_c = env->GetStringUTFChars(jurl, 0);
    std::string url(url_c ? url_c : "");
    env->ReleaseStringUTFChars(jurl, url_c);

    if (url.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "empty url");
        return;
    }

    JavaVM* jvm = nullptr;
    if (env->GetJavaVM(&jvm) != JNI_OK || jvm == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "GetJavaVM failed");
        return;
    }

    jobject managerGlobal = env->NewGlobalRef(thiz);
    g_network_stop.store(false);
    g_network_running.store(true);

    g_network_thread = std::thread([jvm, managerGlobal, url, inputSize, modelid, cpugpu]() {
        JNIEnv* envThread = nullptr;
        if (jvm->AttachCurrentThread(&envThread, nullptr) != JNI_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "AttachCurrentThread failed");
            g_network_running.store(false);
            return;
        }

        jclass managerCls = envThread->GetObjectClass(managerGlobal);
        jmethodID onFrame = envThread->GetMethodID(
                managerCls, "onNetworkFrameReceived",
                "(Landroid/graphics/Bitmap;[Landroid/graphics/RectF;)V");
        jmethodID onError = envThread->GetMethodID(
                managerCls, "onNetworkError",
                "(Ljava/lang/String;)V");
        jmethodID onConnChanged = envThread->GetMethodID(
                managerCls, "onConnectionStatusChanged",
                "(Z)V");
        jmethodID isDetecting = envThread->GetMethodID(
                managerCls, "isDetecting",
                "()Z");

        if (!onFrame || !onError || !onConnChanged || !isDetecting) {
            __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "failed to get callback methods - onFrame:%p onError:%p onConnChanged:%p",
                                onFrame, onError, onConnChanged);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception while getting method IDs");
                envThread->ExceptionDescribe();
                envThread->ExceptionClear();
            }
            g_network_running.store(false);
            jvm->DetachCurrentThread();
            envThread->DeleteGlobalRef(managerGlobal);
            return;
        }

        // 确保模型已加载
        {
            ncnn::MutexLockGuard g(g_lock);
            if (!g_yolo) {
                __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Model not loaded, g_yolo is null");
                jstring msg = envThread->NewStringUTF("模型未加载，请先在Java侧调用 loadModel");
                if (msg) {
                    envThread->CallVoidMethod(managerGlobal, onError, msg);
                    if (envThread->ExceptionCheck()) {
                        __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onError callback");
                        envThread->ExceptionClear();
                    }
                    envThread->DeleteLocalRef(msg);
                }
                envThread->CallVoidMethod(managerGlobal, onConnChanged, JNI_FALSE);
                if (envThread->ExceptionCheck()) {
                    __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onConnectionStatusChanged callback");
                    envThread->ExceptionClear();
                }
                g_network_running.store(false);
                jvm->DetachCurrentThread();
                envThread->DeleteGlobalRef(managerGlobal);
                return;
            }
        }

        FFmpegVideoDecoder decoder;
        __android_log_print(ANDROID_LOG_INFO, "NetworkVideo", "Attempting to connect to: %s", url.c_str());
        if (!decoder.init(url.c_str())) {
            std::string ffmpegError = decoder.getLastError();
            __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Failed to init decoder for url: %s, FFmpeg error: %s",
                                url.c_str(), ffmpegError.c_str());
            // 获取更详细的错误信息，包含FFmpeg的具体错误
            std::string errorMsg = "无法连接网络流\n\nFFmpeg错误: " + ffmpegError +
                                   "\n\n请检查：\n1. 地址是否正确\n2. 网络是否通畅\n3. 服务器是否支持该协议\n4. AndroidManifest.xml是否添加网络权限";
            jstring msg = envThread->NewStringUTF(errorMsg.c_str());
            if (msg) {
                envThread->CallVoidMethod(managerGlobal, onError, msg);
                if (envThread->ExceptionCheck()) {
                    __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onError callback");
                    envThread->ExceptionClear();
                }
                envThread->DeleteLocalRef(msg);
            }
            envThread->CallVoidMethod(managerGlobal, onConnChanged, JNI_FALSE);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onConnectionStatusChanged callback");
                envThread->ExceptionClear();
            }
            g_network_running.store(false);
            jvm->DetachCurrentThread();
            envThread->DeleteGlobalRef(managerGlobal);
            return;
        }
        __android_log_print(ANDROID_LOG_INFO, "NetworkVideo", "Decoder initialized successfully for: %s", url.c_str());

        // 连接成功
        __android_log_print(ANDROID_LOG_INFO, "NetworkVideo", "Successfully connected to: %s", url.c_str());
        envThread->CallVoidMethod(managerGlobal, onConnChanged, JNI_TRUE);
        if (envThread->ExceptionCheck()) {
            __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onConnectionStatusChanged callback");
            envThread->ExceptionClear();
        }

        cv::Mat rgb_frame;
        int frameCount = 0;
        int consecutiveFailures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 10; // 连续失败10次才报错

        while (!g_network_stop.load()) {
            // 必须解码帧（FFmpeg是顺序解码，不能跳过）
            bool decodeSuccess = decoder.decode_frame(rgb_frame);
            if (!decodeSuccess) {
                consecutiveFailures++;
                __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Failed to decode frame, consecutive failures: %d", consecutiveFailures);

                // 如果连续失败次数过多，认为流已断开或无法解码
                if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                    __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Too many consecutive decode failures, stopping stream");
                    jstring msg = envThread->NewStringUTF("无法解码视频流，可能流已断开或格式不支持");
                    if (msg) {
                        envThread->CallVoidMethod(managerGlobal, onError, msg);
                        if (envThread->ExceptionCheck()) {
                            __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onError callback");
                            envThread->ExceptionClear();
                        }
                        envThread->DeleteLocalRef(msg);
                    }
                    break;
                }

                // 短暂等待后重试，避免CPU占用过高
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            // 解码成功，重置失败计数
            consecutiveFailures = 0;

            frameCount++;

            // 固定间隔抽帧：每隔一帧处理一次（对于25FPS视频，处理速度约12.5FPS）
            if (frameCount % 2 == 0) {
                // 跳过这一帧，只解码不处理
                continue;
            }

            // 检查Java层的检测开关状态（添加异常保护）
            jboolean detecting = JNI_FALSE;
            detecting = envThread->CallBooleanMethod(managerGlobal, isDetecting);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception checking isDetecting");
                envThread->ExceptionClear();
                detecting = JNI_FALSE;
            }
            std::vector<Object> objects;
            jobjectArray rectFArray = nullptr;

            // 如果检测启用，进行推理和绘制
            if (detecting) {
                // 开始处理这一帧（推理+回调）
                double t1 = ncnn::get_current_time();
                {
                    ncnn::MutexLockGuard g(g_lock);
                    if (g_yolo) {
                        g_yolo->detect(rgb_frame, objects);
                        drawDetectionsOnFrame(rgb_frame, objects,
                                              g_yolo->getClassNames(),
                                              g_yolo->getColors(),
                                              g_yolo->getClassCount());
                    }
                }

                double t2 = ncnn::get_current_time();
                double allTime = t2 - t1;
                double fps = allTime > 0 ? 1000.0f / allTime : 0.0f;
                float cpu = get_cpu_usage();

                {
                    std::lock_guard<std::mutex> lock(g_summary_mutex);
                    g_summary.allTimeMs = (float)allTime;
                    g_summary.fps = (float)fps;
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

                // 构造 RectF[]，仅用于数量统计
                jclass rectFCls = envThread->FindClass("android/graphics/RectF");
                if (rectFCls) {
                    jmethodID rectCtor = envThread->GetMethodID(rectFCls, "<init>", "(FFFF)V");
                    if (rectCtor && !envThread->ExceptionCheck()) {
                        rectFArray = envThread->NewObjectArray((jsize)objects.size(), rectFCls, nullptr);
                        if (rectFArray && !envThread->ExceptionCheck()) {
                            for (jsize i = 0; i < (jsize)objects.size(); ++i) {
                                const Object& obj = objects[i];
                                float left = obj.rect.x;
                                float top = obj.rect.y;
                                float right = obj.rect.x + obj.rect.width;
                                float bottom = obj.rect.y + obj.rect.height;
                                jobject rect = envThread->NewObject(rectFCls, rectCtor, left, top, right, bottom);
                                if (rect && !envThread->ExceptionCheck()) {
                                    envThread->SetObjectArrayElement(rectFArray, i, rect);
                                    envThread->DeleteLocalRef(rect);
                                } else {
                                    if (envThread->ExceptionCheck()) {
                                        __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception creating RectF object");
                                        envThread->ExceptionClear();
                                    }
                                }
                            }
                        } else {
                            if (envThread->ExceptionCheck()) {
                                __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception creating RectF array");
                                envThread->ExceptionClear();
                            }
                        }
                    } else {
                        if (envThread->ExceptionCheck()) {
                            __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception getting RectF constructor");
                            envThread->ExceptionClear();
                        }
                    }
                    envThread->DeleteLocalRef(rectFCls);
                }
                if (envThread->ExceptionCheck()) {
                    __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception in detection processing");
                    envThread->ExceptionClear();
                    // 检测失败时，创建空的RectF数组，继续显示原始帧
                    jclass rectFCls = envThread->FindClass("android/graphics/RectF");
                    if (rectFCls) {
                        rectFArray = envThread->NewObjectArray(0, rectFCls, nullptr);
                        envThread->DeleteLocalRef(rectFCls);
                    }
                }
            } else {
                // 预览模式：不进行检测，创建空的RectF数组
                jclass rectFCls = envThread->FindClass("android/graphics/RectF");
                if (rectFCls) {
                    rectFArray = envThread->NewObjectArray(0, rectFCls, nullptr);
                    if (envThread->ExceptionCheck()) {
                        __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception creating empty RectF array");
                        envThread->ExceptionClear();
                        rectFArray = nullptr;
                    }
                    envThread->DeleteLocalRef(rectFCls);
                }
            }

            // 无论是否检测，都要回调Java层显示画面
            jobject bitmap = matToBitmap(envThread, rgb_frame);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_WARN, "NetworkVideo", "Exception in matToBitmap");
                envThread->ExceptionClear();
                bitmap = nullptr;
            }
            if (!bitmap) {
                if (rectFArray) {
                    envThread->DeleteLocalRef(rectFArray);
                }
                continue;
            }

            // 回调Java层，添加异常检查
            envThread->CallVoidMethod(managerGlobal, onFrame, bitmap, rectFArray);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onFrame callback");
                envThread->ExceptionClear();
            }

            // 清理资源
            if (bitmap) {
                envThread->DeleteLocalRef(bitmap);
            }
            if (rectFArray) {
                envThread->DeleteLocalRef(rectFArray);
            }

            frameCount++;
        }

        decoder.cleanup();

        // 连接断开（无论是正常停止还是解码失败）
        __android_log_print(ANDROID_LOG_INFO, "NetworkVideo", "Stream stopped, total frames: %d", frameCount);

        // 如果是因为解码失败退出的，已经发送了onError，这里只需要更新连接状态
        if (consecutiveFailures < MAX_CONSECUTIVE_FAILURES) {
            // 正常停止，发送连接断开回调
            envThread->CallVoidMethod(managerGlobal, onConnChanged, JNI_FALSE);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onConnectionStatusChanged callback");
                envThread->ExceptionClear();
            }
        } else {
            // 解码失败退出，已经发送了onError，这里也发送连接断开
            envThread->CallVoidMethod(managerGlobal, onConnChanged, JNI_FALSE);
            if (envThread->ExceptionCheck()) {
                __android_log_print(ANDROID_LOG_ERROR, "NetworkVideo", "Exception in onConnectionStatusChanged callback");
                envThread->ExceptionClear();
            }
        }

        g_network_running.store(false);
        jvm->DetachCurrentThread();
        envThread->DeleteGlobalRef(managerGlobal);
    });
}

extern "C"
JNIEXPORT void JNICALL
Java_com_tencent_CloudDetect_NetworkVideoManager_stopNetworkVideoStream(
        JNIEnv* env, jobject thiz) {
    if (!g_network_running.load()) {
        return;
    }
    g_network_stop.store(true);
    if (g_network_thread.joinable()) {
        g_network_thread.join();
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_tencent_CloudDetect_NetworkStreamActivity_testNetworkConnection(
        JNIEnv* env, jobject thiz, jstring jurl) {
    const char* url_c = env->GetStringUTFChars(jurl, nullptr);
    if (!url_c) {
        return JNI_FALSE;
    }

    std::string url(url_c);
    env->ReleaseStringUTFChars(jurl, url_c);

    if (url.empty()) {
        return JNI_FALSE;
    }

    // 尝试初始化解码器来测试连接
    FFmpegVideoDecoder decoder;
    bool result = decoder.init(url.c_str());

    if (result) {
        // 连接成功，清理资源
        decoder.cleanup();
        __android_log_print(ANDROID_LOG_INFO, "NetworkStream", "连接测试成功: %s", url.c_str());
        return JNI_TRUE;
    } else {
        // 连接失败
        std::string error = decoder.getLastError();
        __android_log_print(ANDROID_LOG_ERROR, "NetworkStream", "连接测试失败: %s, 错误: %s",
                            url.c_str(), error.c_str());
        return JNI_FALSE;
    }
}