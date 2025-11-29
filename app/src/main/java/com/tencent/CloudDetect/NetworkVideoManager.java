package com.tencent.CloudDetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 网络视频流管理器
 * 负责从局域网拉取视频流，解码，推理，并回调结果
 */
public class NetworkVideoManager {
    private static final String TAG = "NetworkVideoManager";

    // 网络视频流状态
    private AtomicBoolean isStreaming = new AtomicBoolean(false);
    private AtomicBoolean isDetecting = new AtomicBoolean(false);

    // 回调接口
    public interface NetworkVideoCallback {
        void onFrame(Bitmap frame, RectF[] boxes);
        void onStreamStarted();
        void onStreamStopped();
        void onError(String error);
        void onConnectionStatusChanged(boolean connected);
    }

    private NetworkVideoCallback callback;
    private Handler mainHandler;
    private Context context;

    // 网络流参数
    private String streamUrl;
    private int inputSize;
    private int modelId;
    private int cpuGpu;

    // 统计信息
    private long frameCount = 0;
    private long startTime = 0;
    private float currentFps = 0;

    public NetworkVideoManager(Context context) {
        this.context = context;
        this.mainHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * 设置回调接口
     */
    public void setCallback(NetworkVideoCallback callback) {
        this.callback = callback;
    }

    /**
     * 开始网络视频流检测
     * @param url 网络视频流URL (支持RTSP://, http://, rtmp://等)
     * @param inputSize 输入尺寸
     * @param modelId 模型ID
     * @param cpuGpu CPU/GPU选择
     */
    public void startNetworkStream(String url, int inputSize, int modelId, int cpuGpu) {
        try {
            if (isStreaming.get()) {
                Log.w(TAG, "网络流已在运行中");
                return;
            }
            if (url == null || url.trim().isEmpty()) {
                    Log.e(TAG, "视频流地址为空");
                    onNetworkError("视频流地址不能为空");
                    return;
            }
            this.streamUrl = url;
            this.inputSize = inputSize;
            this.modelId = modelId;
            this.cpuGpu = cpuGpu;
            Log.i(TAG, "开始网络视频流: " + url);
            // 重置状态
            frameCount = 0;
            startTime = System.currentTimeMillis();

            // 启动网络流检测（默认不检测，只预览）
            isStreaming.set(true);
            isDetecting.set(false); // 默认不检测，只预览

            // 通知回调
            if (callback != null) {
                try {
                    callback.onStreamStarted();
                } catch (Exception e) {
                    Log.e(TAG, "onStreamStarted回调异常", e);
                }
            }
            // 启动JNI层的网络流处理（可能抛出异常）
            try {
                startNetworkVideoStream(url, inputSize, modelId, cpuGpu);
            } catch (Exception e) {
                Log.e(TAG, "JNI调用异常", e);
                isStreaming.set(false);
                isDetecting.set(false);
                onNetworkError("启动视频流失败: " + (e.getMessage() != null ? e.getMessage() : "未知错误"));
            } catch (Throwable t) {
                Log.e(TAG, "JNI调用严重异常", t);
                isStreaming.set(false);
                isDetecting.set(false);
                onNetworkError("启动视频流失败: 系统错误");
            }
        } catch (Exception e) {
            Log.e(TAG, "startNetworkStream异常", e);
            isStreaming.set(false);
            isDetecting.set(false);
            onNetworkError("启动视频流失败: " + (e.getMessage() != null ? e.getMessage() : "未知错误"));
        }
    }

    /**
     * 停止网络视频流
     */
    public void stopNetworkStream() {
        if (!isStreaming.get()) {
            return;
        }

        Log.i(TAG, "停止网络视频流");

        isStreaming.set(false);
        isDetecting.set(false);

        // 停止JNI层的网络流处理
        stopNetworkVideoStream();

        // 计算最终统计信息
        if (startTime > 0) {
            long duration = System.currentTimeMillis() - startTime;
            currentFps = duration > 0 ? (float) frameCount * 1000 / duration : 0;
            Log.i(TAG, String.format("网络流统计: 总帧数=%d, 总时长=%dms, 平均FPS=%.2f",
                    frameCount, duration, currentFps));
        }

        // 通知回调
        if (callback != null) {
            callback.onStreamStopped();
        }
    }

    /**
     * 暂停/恢复检测
     */
    public void setDetectionEnabled(boolean enabled) {
        isDetecting.set(enabled);
        Log.i(TAG, "检测状态: " + (enabled ? "启用" : "暂停"));
    }

    /**
     * 获取当前状态
     */
    public boolean isStreaming() {
        return isStreaming.get();
    }

    public boolean isDetecting() {
        return isDetecting.get();
    }

    public float getCurrentFps() {
        return currentFps;
    }

    public long getFrameCount() {
        return frameCount;
    }

    /**
     * 处理网络流帧回调（由JNI调用）
     */
    public void onNetworkFrameReceived(Bitmap frame, RectF[] boxes) {
        if (!isStreaming.get()) {
            return;
        }
        // 只有检测模式下才统计检测帧数
        if (isDetecting.get()) {
            frameCount++;
            // 计算实时FPS
            long currentTime = System.currentTimeMillis();
            if (startTime > 0) {
                long duration = currentTime - startTime;
                if (duration > 0) {
                    currentFps = (float) frameCount * 1000 / duration;
                }
            }
        }

        // 在主线程中回调（预览模式和检测模式都要回调显示）
        if (callback != null) {
            mainHandler.post(() -> {
                try {
                    callback.onFrame(frame, boxes);
                } catch (Exception e) {
                    Log.e(TAG, "回调处理异常", e);
                }
            });
        }
    }

    /**
     * 处理网络流错误（由JNI调用）
     */
    public void onNetworkError(String error) {
        Log.e(TAG, "网络流错误: " + error);

        isStreaming.set(false);
        isDetecting.set(false);

        if (callback != null) {
            mainHandler.post(() -> {
                try {
                    callback.onError(error);
                } catch (Exception e) {
                    Log.e(TAG, "错误回调处理异常", e);
                }
            });
        }
    }

    /**
     * 处理连接状态变化（由JNI调用）
     */
    public void onConnectionStatusChanged(boolean connected) {
        Log.i(TAG, "连接状态变化: " + (connected ? "已连接" : "已断开"));

        if (callback != null) {
            mainHandler.post(() -> {
                try {
                    callback.onConnectionStatusChanged(connected);
                } catch (Exception e) {
                    Log.e(TAG, "连接状态回调处理异常", e);
                }
            });
        }
    }

    // JNI方法声明
    private native void startNetworkVideoStream(String url, int inputSize, int modelId, int cpuGpu);
    private native void stopNetworkVideoStream();

    // 加载本地库
    static {
        System.loadLibrary("yolov8ncnn");
    }
}
