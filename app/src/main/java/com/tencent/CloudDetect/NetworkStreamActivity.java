package com.tencent.CloudDetect;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.tencent.LocalDetect.R;
import com.tencent.LocalDetect.MainAlg;
import java.util.Locale;

/**
 * 网络视频流配置Activity
 * 负责配置网络流参数，启动和停止网络流检测
 */
public class NetworkStreamActivity extends Activity implements NetworkVideoManager.NetworkVideoCallback {
    private static final String TAG = "NetworkStreamActivity";

    // UI控件
    private EditText etCameraIp;
    private ImageView imageView;
    private SurfaceView surfaceNetworkPreview;
    private TextView tvPreviewPlaceholder;
    private SeekBar seekThreshold, seekNms;
    private TextView tvThresholdValue, tvNmsValue, tvConnectionStatus;
    private TextView tvAllValue, tvInferValue, tvFpsValue, tvCpuValue;
    private TextView tvLog;
    private CheckBox checkboxTrack, checkboxShader;
    private Button btnConnectPreview;
    private Button btnStartDetection, btnStopDetection;

    // 状态管理
    private boolean isConnected = false;
    private boolean isPreviewing = false;
    private boolean isDetecting = false;
    private String currentStreamUrl = "";
    private boolean connectionStatusLocked = false; // 连接状态锁定标志，连接成功后不再更新

    // 网络视频流管理器
    private NetworkVideoManager networkVideoManager;
    private Handler mainHandler;

    // 检测参数
    private float threshold = 0.45f;
    private float nmsThreshold = 0.65f;
    private int timeout = 5000;
    private int bufferSize = 1024;

    // 从WelcomeActivity传递的参数
    private int modelId, inputSize, cpuGpu;
    private MainAlg yolov8ncnn;
    private String modelName = "UnknownModel";
    private String inputSizeLabel = "UnknownSize";
    private String deviceType = "CPU";
    private boolean trackEnabled = true;
    private boolean shaderEnabled = true;

    // 日志刷新
    private final Handler logHandler = new Handler();
    private final Runnable logRefresher = new Runnable() {
        @Override
        public void run() {
            refreshLog();
            logHandler.postDelayed(this, 1000); // 每秒刷新一次
        }
    };

    // 监控信息刷新
    private final Handler monitorHandler = new Handler();
    private long lastCpuTime = 0, lastRealTime = 0;
    private float lastCpuUsage = 0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_network_stream);

        // 获取传递的参数
        Intent intent = getIntent();
        modelId = intent.getIntExtra("model", 0);
        inputSize = intent.getIntExtra("inputsize", 0);
        cpuGpu = intent.getIntExtra("cpugpu", 0);

        // 解析模型名称、输入尺寸和设备类型
        String[] modelNames = getResources().getStringArray(R.array.model_list);
        if (modelId >= 0 && modelId < modelNames.length) {
            modelName = modelNames[modelId];
        }
        String[] inputSizes = getResources().getStringArray(R.array.input_size);
        if (inputSize >= 0 && inputSize < inputSizes.length) {
            inputSizeLabel = inputSizes[inputSize];
        }
        deviceType = (cpuGpu == 0) ? "CPU" : "GPU";

        // 初始化本地JNI推理引擎（与本地检测共用一套模型）
        yolov8ncnn = new MainAlg();
        yolov8ncnn.loadModel(getAssets(), modelId, cpuGpu, inputSize);

        // 初始化
        initViews();
        initListeners();
        initNetworkManager();

        // 初始化推理参数（置信度、NMS、跟踪、轨迹）
        updateInferenceParams();
        Log.i(TAG, String.format("初始化完成 - 模型:%d, 输入尺寸:%d, 设备:%d", modelId, inputSize, cpuGpu));
    }

    /**
     * 初始化UI控件
     */
    private void initViews() {
        etCameraIp = findViewById(R.id.et_camera_ip);
        imageView = findViewById(R.id.imageView);
        surfaceNetworkPreview = findViewById(R.id.surface_network_preview);
        tvPreviewPlaceholder = findViewById(R.id.tv_preview_placeholder);
        seekThreshold = findViewById(R.id.seek_threshold);
        seekNms = findViewById(R.id.seek_nms);
        tvThresholdValue = findViewById(R.id.tv_threshold_value);
        tvNmsValue = findViewById(R.id.tv_nms_value);
        tvConnectionStatus = findViewById(R.id.tv_connection_status);
        tvAllValue = findViewById(R.id.tv_all_value);
        tvInferValue = findViewById(R.id.tv_infer_value);
        tvFpsValue = findViewById(R.id.tv_fps_value);
        tvCpuValue = findViewById(R.id.tv_cpu_value);
        tvLog = findViewById(R.id.tv_log);
        checkboxTrack = findViewById(R.id.checkbox_track);
        checkboxShader = findViewById(R.id.checkbox_shader);
        btnConnectPreview = findViewById(R.id.btn_connect_preview);
        btnStartDetection = findViewById(R.id.btn_start_detection);
        btnStopDetection = findViewById(R.id.btn_stop_detection);

        // 设置初始值
        seekThreshold.setProgress((int)(threshold * 100));
        seekNms.setProgress((int)(nmsThreshold * 100));
        updateThresholdDisplay();
        updateNmsDisplay();

        // Track / Shader 开关默认状态
        if (checkboxTrack != null) {
            checkboxTrack.setChecked(true);
        }
        if (checkboxShader != null) {
            checkboxShader.setChecked(true);
        }

        // 初始状态
        imageView.setVisibility(View.GONE);
        surfaceNetworkPreview.setVisibility(View.GONE);
        tvPreviewPlaceholder.setVisibility(View.VISIBLE);
        resetMonitorValues();

        // 设置SurfaceView回调
        if (surfaceNetworkPreview != null) {
            surfaceNetworkPreview.getHolder().addCallback(new SurfaceHolder.Callback() {
                @Override
                public void surfaceCreated(SurfaceHolder holder) {
                    Log.d(TAG, "Surface已创建");
                }

                @Override
                public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                    Log.d(TAG, String.format("Surface尺寸变化: %dx%d", width, height));
                }

                @Override
                public void surfaceDestroyed(SurfaceHolder holder) {
                    Log.d(TAG, "Surface已销毁");
                }
            });
        }

        // 启动日志刷新和监控刷新
        logHandler.post(logRefresher);
        startMonitorRefresh();
    }

    /**
     * 初始化事件监听器
     */
    private void initListeners() {
        // 置信度阈值调节
        seekThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                threshold = progress / 100.0f;
                updateThresholdDisplay();
                updateInferenceParams(); // 统一更新所有参数
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // NMS阈值调节
        seekNms.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                nmsThreshold = progress / 100.0f;
                updateNmsDisplay();
                updateInferenceParams(); // 统一更新所有参数
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // 连接预览按钮
        btnConnectPreview.setOnClickListener(v -> connectPreview());

        // 开始检测按钮
        btnStartDetection.setOnClickListener(v -> startDetection());

        // 停止检测按钮
        btnStopDetection.setOnClickListener(v -> stopDetection());

        // Track / Shader UI 开关
        if (checkboxTrack != null) {
            checkboxTrack.setOnCheckedChangeListener((buttonView, isChecked) -> {
                trackEnabled = isChecked;
                Log.d(TAG, "Track 开关: " + trackEnabled);
                updateInferenceParams(); // 统一更新所有参数
            });
        }
        if (checkboxShader != null) {
            checkboxShader.setOnCheckedChangeListener((buttonView, isChecked) -> {
                shaderEnabled = isChecked;
                Log.d(TAG, "Shader 开关: " + shaderEnabled);
                updateInferenceParams(); // 统一更新所有参数
            });
        }
    }

    /**
     * 初始化网络视频流管理器
     */
    private void initNetworkManager() {
        networkVideoManager = new NetworkVideoManager(this);
        networkVideoManager.setCallback(this);
        mainHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * 验证URL格式
     * 支持格式: "http://192.168.1.8:4747" 或 "rtsp://192.168.1.2:554/stream" 等
     */
    private boolean isValidUrl(String url) {
        if (url == null || url.trim().isEmpty()) {
            return false;
        }

        url = url.trim().toLowerCase();

        // 检查是否包含协议前缀
        if (!url.startsWith("http://") && !url.startsWith("https://") &&
                !url.startsWith("rtsp://") && !url.startsWith("rtmp://")) {
            return false;
        }

        // 基本URL格式验证
        try {
            java.net.URL urlObj = new java.net.URL(url);
            String host = urlObj.getHost();
            int port = urlObj.getPort();

            // 验证主机地址（可以是IP或域名）
            if (host == null || host.isEmpty()) {
                return false;
            }

            // 如果指定了端口，验证端口范围
            if (port != -1 && (port < 1 || port > 65535)) {
                return false;
            }

            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 从URL中自动识别协议类型
     * @param url 视频流URL
     * @return 协议类型（rtsp, http, https, rtmp等）
     */
    private String detectProtocol(String url) {
        if (url == null || url.trim().isEmpty()) {
            return "rtsp"; // 默认
        }

        url = url.trim().toLowerCase();
        if (url.startsWith("rtsp://")) {
            return "rtsp";
        } else if (url.startsWith("http://")) {
            return "http";
        } else if (url.startsWith("https://")) {
            return "https";
        } else if (url.startsWith("rtmp://")) {
            return "rtmp";
        }

        return "rtsp"; // 默认
    }

    /**
     * 连接预览 - 直接启动预览，通过回调判断连接是否成功
     */
    private void connectPreview() {
        String streamUrl = etCameraIp.getText().toString().trim();
        if (streamUrl.isEmpty()) {
            Toast.makeText(this, "请输入视频流地址", Toast.LENGTH_SHORT).show();
            return;
        }

        // 验证URL格式
        if (!isValidUrl(streamUrl)) {
            Toast.makeText(this, "请输入有效的视频流地址，支持http://或rtsp://协议", Toast.LENGTH_LONG).show();
            return;
        }

        // 确保模型已加载
        if (yolov8ncnn == null) {
            Log.e(TAG, "模型未初始化");
            Toast.makeText(this, "模型未初始化，请重新进入界面", Toast.LENGTH_LONG).show();
            return;
        }

        // 自动识别协议类型
        String protocol = detectProtocol(streamUrl);
        Log.i(TAG, String.format("检测到协议类型: %s", protocol));

        // 更新UI状态
        btnConnectPreview.setEnabled(false);
        tvConnectionStatus.setText("状态: 正在连接...");
        tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
        tvPreviewPlaceholder.setText("正在连接...");

        // 保存URL，用于后续启动
        currentStreamUrl = streamUrl;

        // 重置连接状态锁定
        connectionStatusLocked = false;

        // 直接启动网络流预览，通过回调判断连接是否成功
        try {
            // 启动网络流预览（默认不检测，只预览）
            // 连接成功会回调onStreamStarted，失败会回调onError
            networkVideoManager.startNetworkStream(streamUrl, inputSize, modelId, cpuGpu);
            // 确保检测是关闭的（预览模式）
            networkVideoManager.setDetectionEnabled(false);
            Log.i(TAG, "尝试启动预览模式");
        } catch (Exception e) {
            Log.e(TAG, "启动预览失败", e);
            // 确保UI状态恢复
            handleConnectionError("启动预览失败: " + (e.getMessage() != null ? e.getMessage() : "未知错误"));
        } catch (Throwable t) {
            Log.e(TAG, "启动预览严重异常", t);
            // 捕获所有异常，包括JNI层的异常
            handleConnectionError("启动预览失败: 系统错误");
        }
    }

    /**
     * 开始检测
     */
    private void startDetection() {
        if (!isPreviewing) {
            Toast.makeText(this, "请先连接预览", Toast.LENGTH_SHORT).show();
            return;
        }

        if (isDetecting) {
            Toast.makeText(this, "检测已启动", Toast.LENGTH_SHORT).show();
            return;
        }
        // 确保推理参数已更新（置信度、NMS、跟踪、轨迹）
        updateInferenceParams();

        // 启动检测模式
        if (networkVideoManager != null) {
            networkVideoManager.setDetectionEnabled(true);
            isDetecting = true;
            updateButtonStates();
            startMonitorRefresh();
            startLogRefresh();
            Toast.makeText(this, "已开始检测", Toast.LENGTH_SHORT).show();
            Log.i(TAG, "启动检测模式");
        }
    }

    /**
     * 停止检测 - 返回预览状态
     */
    private void stopDetection() {
        if (!isDetecting) {
            return;
        }

        // 停止检测，返回预览状态（在主线程中执行，避免卡顿）
        mainHandler.post(() -> {
            if (networkVideoManager != null) {
                networkVideoManager.setDetectionEnabled(false);
                isDetecting = false;
                updateButtonStates();
                stopMonitorRefresh();
                stopLogRefresh();
                Toast.makeText(this, "已停止检测，返回预览状态", Toast.LENGTH_SHORT).show();
                Log.i(TAG, "停止检测，返回预览模式");
            }
        });
    }

    /**
     * 更新按钮状态
     */
    private void updateButtonStates() {
        btnStartDetection.setEnabled(isPreviewing && !isDetecting);
        btnStopDetection.setEnabled(isDetecting);
    }

    /**
     * 刷新日志区
     */
    private void refreshLog() {
        if (tvLog == null || yolov8ncnn == null) {
            return;
        }

        String logHeader = String.format(Locale.ENGLISH, "[%s] %s, input_size: %s\n",
                deviceType, modelName, inputSizeLabel);

        if (isDetecting) {
            // 检测模式下显示检测结果
            com.tencent.LocalDetect.DetectSummary summary = yolov8ncnn.getDetectSummary();
            if (summary != null && summary.logText != null && !summary.logText.trim().isEmpty()) {
                tvLog.setText(logHeader + summary.logText);
            } else {
                tvLog.setText(logHeader + "Detect_Info: 0 target");
            }
        } else if (isPreviewing) {
            // 预览模式下显示提示
            tvLog.setText(logHeader + "预览模式：等待检测...");
        } else {
            // 未连接状态
            tvLog.setText(logHeader + "等待连接...");
        }
    }

    /**
     * 重置监控值
     */
    private void resetMonitorValues() {
        if (tvAllValue != null) tvAllValue.setText("--");
        if (tvInferValue != null) tvInferValue.setText("--");
        if (tvFpsValue != null) tvFpsValue.setText("--");
        if (tvCpuValue != null) tvCpuValue.setText("--");
    }

    /**
     * 启动监控刷新
     */
    private void startMonitorRefresh() {
        monitorHandler.removeCallbacks(monitorRefresher);
        lastCpuTime = Process.getElapsedCpuTime();
        lastRealTime = SystemClock.elapsedRealtime();
        lastCpuUsage = 0f;
        monitorHandler.post(monitorRefresher);
    }

    /**
     * 停止监控刷新
     */
    private void stopMonitorRefresh() {
        monitorHandler.removeCallbacks(monitorRefresher);
    }
    /**
     * 启动日志刷新
     */
    private void startLogRefresh() {
        logHandler.removeCallbacks(logRefresher);
        logHandler.post(logRefresher);
    }

    /**
     * 停止日志刷新
     */
    private void stopLogRefresh() {
        logHandler.removeCallbacks(logRefresher);
    }

    /**
     * 监控刷新器
     */
    private final Runnable monitorRefresher = new Runnable() {
        @Override
        public void run() {
            long cpuTime = Process.getElapsedCpuTime();
            long realTime = SystemClock.elapsedRealtime();
            if (lastRealTime > 0) {
                long cpuDelta = cpuTime - lastCpuTime;
                long realDelta = realTime - lastRealTime;
                if (realDelta > 0) {
                    lastCpuUsage = 100f * cpuDelta / (realDelta * 10);
                }
            }
            lastCpuTime = cpuTime;
            lastRealTime = realTime;

            // 更新监控信息
            if (isPreviewing && yolov8ncnn != null) {
                com.tencent.LocalDetect.DetectSummary summary = yolov8ncnn.getDetectSummary();
                if (summary != null) {
                    if (tvAllValue != null) {
                        tvAllValue.setText(String.format("%.0fms", summary.allTimeMs));
                    }
                    if (tvInferValue != null) {
                        tvInferValue.setText(String.format("%.0fms", summary.inferTimeMs));
                    }
                    if (tvFpsValue != null) {
                        tvFpsValue.setText(String.format("%.1f", summary.fps));
                    }
                    if (tvCpuValue != null) {
                        tvCpuValue.setText(String.format("%.1f%%", lastCpuUsage));
                    }
                }
            }

            monitorHandler.postDelayed(this, 1000);
        }
    };



    // 加载本地库
    static {
        System.loadLibrary("yolov8ncnn");
    }

    /**
     * 更新置信度显示
     */
    private void updateThresholdDisplay() {
        tvThresholdValue.setText(String.format(Locale.ENGLISH, "%.2f", threshold));
    }

    /**
     * 更新NMS显示
     */
    private void updateNmsDisplay() {
        tvNmsValue.setText(String.format(Locale.ENGLISH, "%.2f", nmsThreshold));
    }
    /**
     * 更新推理参数（置信度、NMS、跟踪、轨迹）
     * 参考本地视频的实现方式
     */
    private void updateInferenceParams() {
        if (yolov8ncnn != null) {
            yolov8ncnn.setThreshold(threshold);
            yolov8ncnn.setNms(nmsThreshold);
            yolov8ncnn.setTrackEnabled(trackEnabled);
            yolov8ncnn.setShaderEnabled(shaderEnabled);
            Log.d(TAG, String.format("更新推理参数 - 置信度:%.2f, NMS:%.2f, Track:%s, Shader:%s",
                    threshold, nmsThreshold, trackEnabled, shaderEnabled));
        }
    }

    // NetworkVideoCallback 实现
    private boolean firstFrameReceived = false;

    @Override
    public void onFrame(Bitmap frame, RectF[] boxes) {
        // 显示预览画面
        mainHandler.post(() -> {
            try {
                if (frame != null && !frame.isRecycled()) {
                    imageView.setImageBitmap(frame);
                    imageView.setVisibility(View.VISIBLE);
                    surfaceNetworkPreview.setVisibility(View.GONE);
                    tvPreviewPlaceholder.setVisibility(View.GONE);

                    // 第一帧到达时显示成功提示
                    if (!firstFrameReceived) {
                        firstFrameReceived = true;
                        Toast.makeText(this, "预览画面已启动", Toast.LENGTH_SHORT).show();
                        Log.i(TAG, "收到第一帧，视频流正常");
                    }
                }
                Log.d(TAG, "收到网络流帧: " + (frame != null ? frame.getWidth() + "x" + frame.getHeight() : "null"));
            } catch (Exception e) {
                Log.e(TAG, "处理帧回调异常", e);
            }
        });
    }

    @Override
    public void onStreamStarted() {
        mainHandler.post(() -> {
            isConnected = true;
            isPreviewing = true;
            // 确保检测是关闭的（预览模式）
            if (networkVideoManager != null) {
                networkVideoManager.setDetectionEnabled(false);
            }
            // 锁定连接状态，连接成功后不再更新
            connectionStatusLocked = true;
            tvConnectionStatus.setText("状态: 已连接");
            tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            btnConnectPreview.setEnabled(true);
            updateButtonStates();
            // 不立即显示成功提示，等待第一帧到达
            Log.i(TAG, "网络流已启动，等待视频帧...");
        });
    }

    @Override
    public void onStreamStopped() {
        mainHandler.post(() -> {
            firstFrameReceived = false; // 重置标志
            isPreviewing = false;
            isDetecting = false;
            isConnected = false;
            // 解锁状态（在检查之前解锁）
            boolean wasLocked = connectionStatusLocked;
            connectionStatusLocked = false;
            // 只有在之前未锁定状态时才更新连接状态
            if (!wasLocked) {
                tvConnectionStatus.setText("状态: 未连接");
                tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            }
            btnConnectPreview.setEnabled(true);
            updateButtonStates();
            imageView.setVisibility(View.GONE);
            surfaceNetworkPreview.setVisibility(View.GONE);
            tvPreviewPlaceholder.setVisibility(View.VISIBLE);
            tvPreviewPlaceholder.setText("等待连接...");
            Log.i(TAG, "网络流已停止");
        });
    }

    /**
     * 处理连接错误（统一错误处理）
     */
    private void handleConnectionError(String error) {
        try {
            isConnected = false;
            isPreviewing = false;
            isDetecting = false;
            connectionStatusLocked = false; // 解锁状态

            // 截断过长的错误信息
            String safeError = error;
            if (safeError != null && safeError.length() > 150) {
                safeError = safeError.substring(0, 150) + "...";
            }
            if (safeError == null) {
                safeError = "连接失败";
            }

            final String finalError = safeError;
            mainHandler.post(() -> {
                try {
                    // 只有在未锁定状态时才更新连接状态
                    if (!connectionStatusLocked) {
                        tvConnectionStatus.setText("状态: 连接失败");
                        tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
                    }
                    btnConnectPreview.setEnabled(true);
                    updateButtonStates();
                    imageView.setVisibility(View.GONE);
                    surfaceNetworkPreview.setVisibility(View.GONE);
                    tvPreviewPlaceholder.setVisibility(View.VISIBLE);
                    tvPreviewPlaceholder.setText("连接失败: " + finalError);
                    Toast.makeText(this, "网络流错误: " + finalError, Toast.LENGTH_LONG).show();
                    Log.e(TAG, "网络流错误: " + finalError);
                } catch (Exception e) {
                    Log.e(TAG, "处理错误回调异常", e);
                    // 即使UI更新失败，也要确保按钮可用
                    try {
                        btnConnectPreview.setEnabled(true);
                    } catch (Exception ex) {
                        Log.e(TAG, "恢复按钮状态失败", ex);
                    }
                }
            });
        } catch (Exception e) {
            Log.e(TAG, "handleConnectionError异常", e);
            // 确保按钮可用
            try {
                mainHandler.post(() -> {
                    try {
                        btnConnectPreview.setEnabled(true);
                    } catch (Exception ex) {
                        Log.e(TAG, "恢复按钮状态失败", ex);
                    }
                });
            } catch (Exception ex) {
                Log.e(TAG, "发送恢复按钮状态消息失败", ex);
            }
        }
    }

    @Override
    public void onError(String error) {
        handleConnectionError(error);
    }

    @Override
    public void onConnectionStatusChanged(boolean connected) {
        mainHandler.post(() -> {
            // 如果连接状态已锁定，不再更新状态显示
            if (connectionStatusLocked) {
                Log.d(TAG, "连接状态已锁定，忽略状态变化: " + connected);
                return;
            }

            if (connected) {
                tvConnectionStatus.setText("状态: 已连接");
                tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            } else {
                isPreviewing = false;
                isDetecting = false;
                tvConnectionStatus.setText("状态: 连接断开");
                tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
            }
            updateButtonStates();
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        logHandler.removeCallbacks(logRefresher);
        stopMonitorRefresh();
        if (networkVideoManager != null) {
            networkVideoManager.stopNetworkStream();
        }
    }
}
