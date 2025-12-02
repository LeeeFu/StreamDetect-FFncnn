package NcnnTencent.common;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;

/**
 * 数据模型类集合
 * 整合所有数据模型和业务模型，简化项目结构
 */
public class Models {

    // ========== 数据模型 ==========

    /**
     * 检测结果摘要
     * 统一的数据模型，用于在JNI层和Java层之间传递推理结果
     */
    public static class DetectSummary {
        public float allTimeMs;
        public float inferTimeMs;
        public float fps;
        public String logText;

        public DetectSummary(float allTimeMs, float inferTimeMs, float fps, String logText) {
            this.allTimeMs = allTimeMs;
            this.inferTimeMs = inferTimeMs;
            this.fps = fps;
            this.logText = logText;
        }

        public DetectSummary() {
            this(0, 0, 0, "");
        }
    }

    // ========== 业务模型 ==========

    /**
     * 相机模型层
     * 封装相机相关的业务逻辑
     */
    public static class CameraModel {
        private JniBridge jniBridge;
        private int facing = 1; // 默认后置摄像头

        public CameraModel(JniBridge jniBridge) {
            this.jniBridge = jniBridge;
        }

        /**
         * 打开相机
         * @param facing 相机朝向 (0=前置, 1=后置)
         * @return 是否打开成功
         */
        public boolean openCamera(int facing) {
            this.facing = facing;
            return jniBridge.openCamera(facing);
        }

        /**
         * 关闭相机
         * @return 是否关闭成功
         */
        public boolean closeCamera() {
            return jniBridge.closeCamera();
        }

        /**
         * 设置输出窗口
         * @param surface Surface对象
         * @return 是否设置成功
         */
        public boolean setOutputWindow(Surface surface) {
            return jniBridge.setOutputWindow(surface);
        }
    }

    /**
     * 推理模型层
     * 封装推理相关的业务逻辑
     */
    public static class InferenceModel {
        private JniBridge jniBridge;
        private InferenceConfig config;
        private AssetManager assetManager;

        public InferenceModel(JniBridge jniBridge, AssetManager assetManager) {
            this.jniBridge = jniBridge;
            this.assetManager = assetManager;
        }

        /**
         * 加载模型
         * @param config 配置
         * @return 是否加载成功
         */
        public boolean loadModel(InferenceConfig config) {
            this.config = config;
            return jniBridge.loadModel(
                    assetManager,
                    config.getModelId(),
                    config.getDeviceType(),
                    config.getInputSize()
            );
        }

        /**
         * 检测图片
         * @param bitmap 输入图片
         * @return 检测结果图片
         */
        public Bitmap detectImage(Bitmap bitmap) {
            int inputSizeInt = parseInputSize(config.getInputSize());
            return jniBridge.detectImage(bitmap, inputSizeInt);
        }

        /**
         * 获取检测摘要
         * @return DetectSummary
         */
        public DetectSummary getDetectSummary() {
            return jniBridge.getDetectSummary();
        }

        /**
         * 更新推理参数
         */
        public void updateInferenceParams() {
            if (config == null) return;
            jniBridge.setThreshold(config.getThreshold());
            jniBridge.setNms(config.getNmsThreshold());
            jniBridge.setTrackEnabled(config.isTrackEnabled());
            jniBridge.setShaderEnabled(config.isShaderEnabled());
        }

        /**
         * 设置检测开关
         * @param enabled 是否启用
         */
        public void setDetectEnabled(boolean enabled) {
            jniBridge.setDetectEnabled(enabled);
        }

        /**
         * 解析输入尺寸
         * @param inputSizeIndex 输入尺寸索引
         * @return 输入尺寸数值
         */
        private int parseInputSize(int inputSizeIndex) {
            // 根据索引返回对应的尺寸值
            // 0=320, 1=640, 2=1280等，根据实际配置调整
            switch (inputSizeIndex) {
                case 0: return 320;
                case 1: return 640;
                case 2: return 1280;
                default: return 320;
            }
        }
    }

    /**
     * 网络流模型层
     * 封装网络视频流相关的业务逻辑
     */
    public static class NetworkModel {
        private JniBridge jniBridge;
        private String currentStreamUrl;
        private boolean isStreaming = false;

        public NetworkModel(JniBridge jniBridge, AssetManager assetManager) {
            this.jniBridge = jniBridge;
        }
        /**
         * 启动网络视频流
         * @param url 流地址
         * @param config 配置
         * @param callback 回调对象（NetworkVideoManager实例）
         * @return 是否启动成功
         */
        public boolean startNetworkStream(String url, InferenceConfig config, Object callback) {
            if (isStreaming) {
                return false;
            }
            this.currentStreamUrl = url;
            int inputSizeInt = parseInputSize(config.getInputSize());
            jniBridge.startNetworkVideoStream(
                    url,
                    inputSizeInt,
                    config.getModelId(),
                    config.getDeviceType(),
                    callback
            );
            isStreaming = true;
            return true;
        }

        /**
         * 停止网络视频流
         */
        public void stopNetworkStream() {
            if (!isStreaming) {
                return;
            }
            jniBridge.stopNetworkVideoStream();
            isStreaming = false;
            currentStreamUrl = null;
        }

        /**
         * 是否正在流式传输
         * @return 是否正在流式传输
         */
        public boolean isStreaming() {
            return isStreaming;
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
    }
}

