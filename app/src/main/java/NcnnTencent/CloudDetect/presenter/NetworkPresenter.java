package NcnnTencent.CloudDetect.presenter;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import NcnnTencent.common.InferenceConfig;
import NcnnTencent.common.JniBridge;
import NcnnTencent.common.Models.DetectSummary;
import NcnnTencent.common.Models.InferenceModel;
import NcnnTencent.common.Models.NetworkModel;

/**
 * 网络检测Presenter
 * 处理网络视频流检测的业务逻辑
 */
public class NetworkPresenter {
    public interface View {
        void showFrame(Bitmap frame, RectF[] boxes);
        void showStatusMessage(String message);
        void showError(String error);
        void updateConnectionStatus(boolean connected);
        void updateMonitor(DetectSummary summary);
        void onStreamStarted();
        void onStreamStopped();
    }

    private View view;
    private Context context;
    private JniBridge jniBridge;
    private InferenceModel inferenceModel;
    private NetworkModel networkModel;
    private InferenceConfig config;
    private boolean isDetecting = false;

    public NetworkPresenter(View view, Context context, JniBridge jniBridge) {
        this.view = view;
        this.context = context;
        this.jniBridge = jniBridge;
        this.inferenceModel = new InferenceModel(jniBridge, context.getAssets());
        this.networkModel = new NetworkModel(jniBridge, context.getAssets());
    }

    /**
     * 初始化配置
     */
    public void initializeConfig(int modelId, int inputSize, int deviceType) {
        this.config = new InferenceConfig(modelId, inputSize, deviceType);
        boolean success = inferenceModel.loadModel(config);
        if (!success) {
            view.showError("模型加载失败");
        }
    }

    /**
     * 启动网络流
     * @param url 流地址
     * @param callback 回调对象（用于JNI回调，通常是NetworkVideoManager实例）
     */
    public void startNetworkStream(String url, Object callback) {
        if (url == null || url.trim().isEmpty()) {
            view.showError("视频流地址不能为空");
            return;
        }

        if (networkModel.isStreaming()) {
            view.showStatusMessage("网络流已在运行中");
            return;
        }

        boolean success = networkModel.startNetworkStream(url, config, callback);
        if (success) {
            view.onStreamStarted();
            view.showStatusMessage("网络流已启动");
        } else {
            view.showError("启动网络流失败");
        }
    }

    /**
     * 停止网络流
     */
    public void stopNetworkStream() {
        if (!networkModel.isStreaming()) {
            return;
        }
        networkModel.stopNetworkStream();
        isDetecting = false;
        view.onStreamStopped();
        view.showStatusMessage("网络流已停止");
    }

    /**
     * 切换检测状态
     */
    public void toggleDetection() {
        isDetecting = !isDetecting;
        inferenceModel.setDetectEnabled(isDetecting);
        view.showStatusMessage(isDetecting ? "已开启检测" : "已暂停检测");
    }

    /**
     * 更新推理参数
     */
    public void updateInferenceParams(float threshold, float nms, boolean trackEnabled, boolean shaderEnabled) {
        if (config == null) return;
        config.setThreshold(threshold);
        config.setNmsThreshold(nms);
        config.setTrackEnabled(trackEnabled);
        config.setShaderEnabled(shaderEnabled);
        inferenceModel.updateInferenceParams();
    }

    /**
     * 处理网络帧回调（由JNI调用）
     */
    public void onNetworkFrameReceived(Bitmap frame, RectF[] boxes) {
        if (isDetecting) {
            DetectSummary summary = inferenceModel.getDetectSummary();
            if (summary != null) {
                view.updateMonitor(summary);
            }
        }
        view.showFrame(frame, boxes);
    }

    /**
     * 处理连接状态变化（由JNI调用）
     */
    public void onConnectionStatusChanged(boolean connected) {
        view.updateConnectionStatus(connected);
    }

    /**
     * 处理错误（由JNI调用）
     */
    public void onNetworkError(String error) {
        networkModel.stopNetworkStream();
        isDetecting = false;
        view.showError(error);
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        stopNetworkStream();
    }

    public boolean isStreaming() {
        return networkModel.isStreaming();
    }

    public boolean isDetecting() {
        return isDetecting;
    }
}

