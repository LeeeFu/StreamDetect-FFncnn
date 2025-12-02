package NcnnTencent.common;

/**
 * 推理配置类
 * 统一管理模型、输入尺寸、设备类型等配置参数
 */
public class InferenceConfig {
    private int modelId;
    private int inputSize;
    private int deviceType; // 0=CPU, 1=GPU
    private float threshold;
    private float nmsThreshold;
    private boolean trackEnabled;
    private boolean shaderEnabled;

    public InferenceConfig() {
        // 默认值
        this.modelId = 0;
        this.inputSize = 0;
        this.deviceType = 0;
        this.threshold = 0.45f;
        this.nmsThreshold = 0.65f;
        this.trackEnabled = false;
        this.shaderEnabled = false;
    }
    public InferenceConfig(int modelId, int inputSize, int deviceType) {
        this();
        this.modelId = modelId;
        this.inputSize = inputSize;
        this.deviceType = deviceType;
    }

    // Getters and Setters
    public int getModelId() {
        return modelId;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public int getDeviceType() {
        return deviceType;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    public float getNmsThreshold() {
        return nmsThreshold;
    }

    public void setNmsThreshold(float nmsThreshold) {
        this.nmsThreshold = nmsThreshold;
    }

    public boolean isTrackEnabled() {
        return trackEnabled;
    }

    public void setTrackEnabled(boolean trackEnabled) {
        this.trackEnabled = trackEnabled;
    }

    public boolean isShaderEnabled() {
        return shaderEnabled;
    }

    public void setShaderEnabled(boolean shaderEnabled) {
        this.shaderEnabled = shaderEnabled;
    }

    public boolean isGpuMode() {
        return deviceType == 1;
    }

    public String getDeviceTypeString() {
        return deviceType == 0 ? "CPU" : "GPU";
    }
}

