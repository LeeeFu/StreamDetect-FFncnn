package NcnnTencent.common;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;
import NcnnTencent.common.Models.DetectSummary;

/**
 * JNI桥接层
 * 统一封装所有JNI调用，提供类型安全的接口
 */
public class JniBridge {
    static {
        System.loadLibrary("yolov8ncnn");
    }

    // ========== 模型相关 ==========
    /**
     * 加载模型
     * @param assetManager AssetManager
     * @param modelId 模型ID
     * @param deviceType 设备类型 (0=CPU, 1=GPU)
     * @param inputSize 输入尺寸索引
     * @return 是否加载成功
     * JNI方法签名：Java_com_tencent_common_JniBridge_loadModel
     */
    public native boolean loadModel(AssetManager assetManager, int modelId, int deviceType, int inputSize);

    // ========== 相机相关 ==========
    /**
     * 打开相机
     * @param facing 相机朝向 (0=前置, 1=后置)
     * @return 是否打开成功
     * JNI方法签名：Java_com_tencent_common_JniBridge_openCamera
     */
    public native boolean openCamera(int facing);

    /**
     * 关闭相机
     * @return 是否关闭成功
     * JNI方法签名：Java_com_tencent_common_JniBridge_closeCamera
     */
    public native boolean closeCamera();

    /**
     * 设置输出窗口
     * @param surface Surface对象
     * @return 是否设置成功
     * JNI方法签名：Java_com_tencent_common_JniBridge_setOutputWindow
     */
    public native boolean setOutputWindow(Surface surface);

    // ========== 推理参数设置 ==========
    /**
     * 设置置信度阈值
     * @param threshold 阈值 (0.0-1.0)
     * JNI方法签名：Java_com_tencent_common_JniBridge_setThreshold
     */
    public native void setThreshold(float threshold);

    /**
     * 设置NMS阈值
     * @param nms NMS阈值 (0.0-1.0)
     * JNI方法签名：Java_com_tencent_common_JniBridge_setNms
     */
    public native void setNms(float nms);

    /**
     * 设置跟踪开关
     * @param enabled 是否启用跟踪
     * JNI方法签名：Java_com_tencent_common_JniBridge_setTrackEnabled
     */
    public native void setTrackEnabled(boolean enabled);

    /**
     * 设置Shader拖尾开关
     * @param enabled 是否启用Shader拖尾
     * JNI方法签名：Java_com_tencent_common_JniBridge_setShaderEnabled
     */
    public native void setShaderEnabled(boolean enabled);

    /**
     * 设置检测开关
     * @param enabled 是否启用检测
     * JNI方法签名：Java_com_tencent_common_JniBridge_setDetectEnabled
     */
    public native void setDetectEnabled(boolean enabled);

    // ========== 推理相关 ==========
    /**
     * 检测图片
     * @param bitmap 输入图片
     * @param inputSize 输入尺寸
     * @return 检测结果图片
     * JNI方法签名：Java_com_tencent_common_JniBridge_detectImage
     */
    public native Bitmap detectImage(Bitmap bitmap, int inputSize);

    /**
     * 获取检测摘要
     * @return DetectSummary对象
     * JNI方法签名：Java_com_tencent_common_JniBridge_getDetectSummary
     */
    public native DetectSummary getDetectSummary();

    // ========== 视频检测相关 ==========
    /**
     * 启动FFmpeg视频检测
     * @param videoPath 视频路径
     * @param inputSize 输入尺寸
     * @param callback 回调接口
     * @param assetManager AssetManager
     * @param modelId 模型ID
     * @param deviceType 设备类型
     * JNI方法签名：Java_com_tencent_common_JniBridge_startFFmpegVideoDetect
     */
    public native void startFFmpegVideoDetect(
            String videoPath,
            int inputSize,
            FFmpegDetectCallback callback,
            AssetManager assetManager,
            int modelId,
            int deviceType
    );

    // ========== 网络流相关 ==========
    /**
     * 启动网络视频流
     * @param url 流地址
     * @param inputSize 输入尺寸
     * @param modelId 模型ID
     * @param deviceType 设备类型
     * @param callback 回调对象（NetworkVideoManager实例）
     * JNI方法签名：Java_com_tencent_common_JniBridge_startNetworkVideoStream
     */
    public native void startNetworkVideoStream(String url, int inputSize, int modelId, int deviceType, Object callback);

    /**
     * 停止网络视频流
     * JNI方法签名：Java_com_tencent_common_JniBridge_stopNetworkVideoStream
     */
    public native void stopNetworkVideoStream();

    // ========== 回调接口 ==========
    /**
     * FFmpeg视频检测回调接口
     */
    public interface FFmpegDetectCallback {
        void onFrame(Bitmap frame, android.graphics.RectF[] boxes);
        void onFinish();
        void onError(String msg);
    }
}

