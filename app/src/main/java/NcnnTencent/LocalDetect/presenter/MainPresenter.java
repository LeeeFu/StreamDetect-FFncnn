package NcnnTencent.LocalDetect.presenter;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.view.Surface;
import NcnnTencent.common.InferenceConfig;
import NcnnTencent.common.JniBridge;
import NcnnTencent.common.Models.CameraModel;
import NcnnTencent.common.Models.DetectSummary;
import NcnnTencent.common.Models.InferenceModel;
import NcnnTencent.common.ImageUtils;

import java.io.File;

/**
 * MainLocalActivity的Presenter
 * 处理本地检测的业务逻辑
 */
public class MainPresenter {
    public interface View {
        void showImageResult(Bitmap bitmap);
        void showStatusMessage(String message);
        void showError(String error);
        void updateMonitor(DetectSummary summary);
        void switchToImageView();
        void switchToCameraView();
        void startMonitoring(); // 启动监控
        void stopMonitoring(); // 停止监控
    }

    private View view;
    private Context context;
    private JniBridge jniBridge;
    private InferenceModel inferenceModel;
    private CameraModel cameraModel;
    private InferenceConfig config;
    private boolean isCameraDetecting = false;
    private boolean isVideoDetecting = false;
    private boolean isImageDetecting = false;

    public MainPresenter(View view, Context context, JniBridge jniBridge) {
        this.view = view;
        this.context = context;
        this.jniBridge = jniBridge;
        this.inferenceModel = new InferenceModel(jniBridge, context.getAssets());
        this.cameraModel = new CameraModel(jniBridge);
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
     * 处理图片检测
     */
    public void handleImageDetection(Uri imageUri) {
        if (isImageDetecting || isVideoDetecting) {
            view.showStatusMessage("检测正在进行中，请稍候");
            return;
        }

        Bitmap bitmap = ImageUtils.getPictureFromUri(context, imageUri);
        if (bitmap == null) {
            view.showError("无法读取图片");
            return;
        }

        isImageDetecting = true;
        view.switchToImageView();
        view.showStatusMessage("开始图片检测");

        new Thread(() -> {
            try {
                cameraModel.closeCamera();
                Bitmap resultBitmap = inferenceModel.detectImage(bitmap);
                DetectSummary summary = inferenceModel.getDetectSummary();
                view.showImageResult(resultBitmap);
                if (summary != null) {
                    view.updateMonitor(summary);
                }
            } catch (Exception e) {
                view.showError("图片检测失败: " + e.getMessage());
            } finally {
                isImageDetecting = false;
            }
        }).start();
    }

    /**
     * 处理视频检测
     * 注意：由于JNI方法签名绑定到MainLocalActivity，需要通过View接口回调
     */
    public void handleVideoDetection(Uri videoUri, VideoDetectCallback callback) {
        if (isVideoDetecting || isImageDetecting) {
            view.showStatusMessage("检测正在进行中，请稍候");
            return;
        }

        String videoPath = ImageUtils.getRealPathFromUri(context, videoUri);
        File file = new File(videoPath);
        if (!file.exists() || !file.canRead()) {
            view.showError("视频文件不存在或无法读取");
            return;
        }

        String lowerPath = videoPath.toLowerCase();
        if (!(lowerPath.endsWith(".mp4") || lowerPath.endsWith(".avi") ||
                lowerPath.endsWith(".mov") || lowerPath.endsWith(".mkv"))) {
            view.showError("仅支持mp4/avi/mov/mkv格式");
            return;
        }

        isVideoDetecting = true;
        cameraModel.closeCamera();
        view.switchToImageView();
        view.showStatusMessage("开始视频检测");
        view.startMonitoring(); // 启动监控

        int inputSizeInt = parseInputSize(config.getInputSize());

        // 通过View接口调用native方法（因为JNI签名绑定到Activity）
        if (callback != null) {
            callback.startVideoDetect(
                    videoPath,
                    inputSizeInt,
                    context.getAssets(),
                    config.getModelId(),
                    config.getDeviceType()
            );
        }
    }

    /**
     * 视频检测回调接口
     * 用于在View层实现native方法调用
     */
    public interface VideoDetectCallback {
        void startVideoDetect(String videoPath, int inputSize,
                              android.content.res.AssetManager assetManager,
                              int modelId, int deviceType);
    }

    /**
     * 处理视频检测帧回调
     */
    public void onVideoFrameReceived(Bitmap frame, RectF[] boxes) {
        DetectSummary summary = inferenceModel.getDetectSummary();
        if (summary != null) {
            view.updateMonitor(summary);
        }
        view.showImageResult(frame);
    }

    /**
     * 处理视频检测完成
     */
    public void onVideoDetectFinish() {
        isVideoDetecting = false;
        view.stopMonitoring(); // 停止监控
        view.switchToCameraView();
        view.showStatusMessage("视频检测完成");
    }

    /**
     * 处理视频检测错误
     */
    public void onVideoDetectError(String msg) {
        isVideoDetecting = false;
        view.stopMonitoring(); // 停止监控
        view.showError("视频检测失败: " + msg);
        view.switchToCameraView();
    }

    /**
     * 切换相机检测状态
     */
    public void toggleCameraDetection() {
        if (!isCameraDetecting) {
            inferenceModel.setDetectEnabled(true);
            isCameraDetecting = true;
            view.showStatusMessage("已开启摄像头检测");
            view.startMonitoring(); // 启动监控
        } else {
            inferenceModel.setDetectEnabled(false);
            isCameraDetecting = false;
            view.showStatusMessage("已关闭摄像头检测");
            view.stopMonitoring(); // 停止监控
        }
    }

    /**
     * 处理Surface创建
     */
    public void onSurfaceCreated(Surface surface) {
        cameraModel.setOutputWindow(surface);
        cameraModel.openCamera(1); // 默认后置
        inferenceModel.setDetectEnabled(false); // 默认只预览
    }

    /**
     * 处理Surface变化
     */
    public void onSurfaceChanged(Surface surface) {
        cameraModel.setOutputWindow(surface);
    }

    /**
     * 处理Surface销毁
     */
    public void onSurfaceDestroyed() {
        cameraModel.closeCamera();
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        cameraModel.closeCamera();
        isVideoDetecting = false;
        isImageDetecting = false;
        isCameraDetecting = false;
    }

    /**
     * 解析输入尺寸
     */
    private int parseInputSize(int inputSizeIndex) {
        switch (inputSizeIndex) {
            case 0: return 320;
            case 1: return 640;
            case 2: return 1280;
            default: return 320;
        }
    }

    public boolean isCameraDetecting() {
        return isCameraDetecting;
    }

    public boolean isVideoDetecting() {
        return isVideoDetecting;
    }
}

