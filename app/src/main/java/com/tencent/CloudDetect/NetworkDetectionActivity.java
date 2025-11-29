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
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.tencent.LocalDetect.R;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * 网络视频流检测Activity
 * 用于显示实时检测结果和控制检测过程
 */
public class NetworkDetectionActivity extends Activity implements SurfaceHolder.Callback, NetworkVideoManager.NetworkVideoCallback {
    private static final String TAG = "NetworkDetectionActivity";
    
    // UI控件
    private SurfaceView surfaceNetworkPreview;
    private ImageView imageView;
    private TextView tvConnectionStatus, tvFps, tvFrameCount, tvDetectionCount;
    private TextView tvAllValue, tvInferValue, tvCpuValue, tvLog;
    private Button btnPauseResume, btnScreenshot, btnStopStream, btnBackToConfig;
    private SeekBar seekThreshold, seekNms;
    private TextView tvThresholdValue, tvNmsValue;
    private View overlayDetectionInfo;
    private TextView tvDetectionInfo;

    // 帧率控制相关
    private final Object frameLock = new Object();
    private Bitmap lastFrameBitmap = null;
    
    // 网络视频流管理器
    private NetworkVideoManager networkVideoManager;
    private Handler mainHandler;
    
    // 检测参数
    private float threshold = 0.45f;
    private float nmsThreshold = 0.65f;
    
    // 状态管理
    private boolean isPaused = false;
    private boolean isStreaming = false;
    
    // 统计信息
    private long totalFrameCount = 0;
    private long totalDetectionCount = 0;
    private float currentFps = 0;

    // 监控信息刷新
    private final Handler monitorHandler = new Handler();
    private long lastCpuTime = 0, lastRealTime = 0;
    private float lastCpuUsage = 0f;
    
    // 从配置界面传递的参数
    private String streamUrl;
    private int inputSize, modelId, cpuGpu;

    // 优化UI更新
    private OptimizedUIUpdater uiUpdater;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_network_detection);
        
        // 获取传递的参数
        Intent intent = getIntent();
        streamUrl = intent.getStringExtra("stream_url");
        inputSize = intent.getIntExtra("input_size", 0);
        modelId = intent.getIntExtra("model_id", 0);
        cpuGpu = intent.getIntExtra("cpu_gpu", 0);
        threshold = intent.getFloatExtra("threshold", 0.45f);
        nmsThreshold = intent.getFloatExtra("nms_threshold", 0.65f);
        
        // 初始化
        initViews();
        initListeners();
        initNetworkManager();
        startNetworkStream();
        
        Log.i(TAG, String.format("初始化完成 - URL:%s, 输入尺寸:%d, 模型:%d, 设备:%d", 
                streamUrl, inputSize, modelId, cpuGpu));
    }
    
    /**
     * 初始化UI控件
     */
    private void initViews() {
        surfaceNetworkPreview = findViewById(R.id.surface_network_preview);
        imageView = findViewById(R.id.imageView);
        if (imageView == null) {
            // 如果布局中没有ImageView，创建一个
            imageView = new ImageView(this);
        }
        tvConnectionStatus = findViewById(R.id.tv_connection_status);
        tvFps = findViewById(R.id.tv_fps);
        tvFrameCount = findViewById(R.id.tv_frame_count);
        tvDetectionCount = findViewById(R.id.tv_detection_count);
        tvAllValue = findViewById(R.id.tv_all_value);
        tvInferValue = findViewById(R.id.tv_infer_value);
        tvCpuValue = findViewById(R.id.tv_cpu_value);
        tvLog = findViewById(R.id.tv_log);
        btnPauseResume = findViewById(R.id.btn_pause_resume);
        btnScreenshot = findViewById(R.id.btn_screenshot);
        btnStopStream = findViewById(R.id.btn_stop_stream);
        btnBackToConfig = findViewById(R.id.btn_back_to_config);
        seekThreshold = findViewById(R.id.seek_threshold);
        seekNms = findViewById(R.id.seek_nms);
        tvThresholdValue = findViewById(R.id.tv_threshold_value);
        tvNmsValue = findViewById(R.id.tv_nms_value);
        overlayDetectionInfo = findViewById(R.id.overlay_detection_info);
        tvDetectionInfo = findViewById(R.id.tv_detection_info);

        // 设置SurfaceView回调
        if (surfaceNetworkPreview != null) {
            surfaceNetworkPreview.getHolder().addCallback(this);
        }
        if (seekThreshold != null) {
            seekThreshold.setProgress((int)(threshold * 100));
        }
        if (seekNms != null) {
            seekNms.setProgress((int)(nmsThreshold * 100));
        }
        updateThresholdDisplay();
        updateNmsDisplay();

        // 初始化UI更新器
        uiUpdater = new OptimizedUIUpdater(new Handler());

        // 初始化监控信息
        resetMonitorValues();
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
                updateDetectionParams();
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
                updateDetectionParams();
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        
        // 暂停/恢复按钮
        btnPauseResume.setOnClickListener(v -> togglePauseResume());
        
        // 截图按钮
        btnScreenshot.setOnClickListener(v -> takeScreenshot());
        
        // 停止检测按钮
        btnStopStream.setOnClickListener(v -> stopNetworkStream());
        
        // 返回配置按钮
        btnBackToConfig.setOnClickListener(v -> finish());
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
     * 开始网络视频流检测
     */
    private void startNetworkStream() {
        if (streamUrl == null || streamUrl.isEmpty()) {
            Toast.makeText(this, "无效的流地址", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }
        
        networkVideoManager.startNetworkStream(streamUrl, inputSize, modelId, cpuGpu);
        isStreaming = true;
        
        Log.i(TAG, "开始网络流检测: " + streamUrl);
    }
    
    /**
     * 停止网络视频流检测
     */
    private void stopNetworkStream() {
        networkVideoManager.stopNetworkStream();
        isStreaming = false;
        
        Log.i(TAG, "停止网络流检测");
    }
    
    /**
     * 切换暂停/恢复状态
     */
    private void togglePauseResume() {
        isPaused = !isPaused;
        networkVideoManager.setDetectionEnabled(!isPaused);
        
        if (isPaused) {
            btnPauseResume.setText("恢复检测");
            Toast.makeText(this, "检测已暂停", Toast.LENGTH_SHORT).show();
        } else {
            btnPauseResume.setText("暂停检测");
            Toast.makeText(this, "检测已恢复", Toast.LENGTH_SHORT).show();
        }
        
        Log.i(TAG, "检测状态: " + (isPaused ? "暂停" : "恢复"));
    }
    
    /**
     * 截图功能
     */
    private void takeScreenshot() {
        // 这里可以实现截图功能
        // 暂时显示提示
        Toast.makeText(this, "截图功能开发中...", Toast.LENGTH_SHORT).show();
        
        Log.i(TAG, "截图功能调用");
    }
    
    /**
     * 更新检测参数
     */
    private void updateDetectionParams() {
        // 这里可以调用JNI方法更新检测参数
        Log.d(TAG, String.format("更新检测参数 - 置信度:%.2f, NMS:%.2f", threshold, nmsThreshold));
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
     * 更新统计信息显示
     */
    private void updateStatistics() {
        mainHandler.post(() -> {
            if (tvFrameCount != null) {
                tvFrameCount.setText(String.format("帧数: %d", totalFrameCount));
            }
            if (tvDetectionCount != null) {
                tvDetectionCount.setText(String.format("检测: %d", totalDetectionCount));
            }
            if (tvFps != null) {
                tvFps.setText(String.format("FPS: %.1f", currentFps));
            }
        });
    }

    /**
     * 重置监控值
     */
    private void resetMonitorValues() {
        if (tvAllValue != null) tvAllValue.setText("--");
        if (tvInferValue != null) tvInferValue.setText("--");
        if (tvFps != null) tvFps.setText("--");
        if (tvCpuValue != null) tvCpuValue.setText("--");
        lastCpuUsage = 0f;
        lastCpuTime = 0;
        lastRealTime = 0;
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
            if (tvAllValue != null) {
                tvAllValue.setText(String.format("%.0fms", currentFps > 0 ? 1000.0f / currentFps : 0));
            }
            if (tvInferValue != null) {
                tvInferValue.setText(String.format("%.0fms", currentFps > 0 ? 1000.0f / currentFps : 0));
            }
            if (tvFps != null) {
                tvFps.setText(String.format("FPS: %.1f", currentFps));
            }
            if (tvCpuValue != null) {
                tvCpuValue.setText(String.format("%.1f%%", lastCpuUsage));
            }

            monitorHandler.postDelayed(this, 1000);
        }
    };

    /**
     * 优化UI更新器
     */
    private static class OptimizedUIUpdater {
        private final Handler uiHandler;
        private final Object bufferLock = new Object();
        private Bitmap frontBuffer = null;
        private volatile boolean isUpdating = false;

        public OptimizedUIUpdater(Handler handler) {
            this.uiHandler = handler;
        }

        public void updateUI(Bitmap newFrame, ImageView imageView, SurfaceView viewFinder) {
            synchronized (bufferLock) {
                if (frontBuffer != null && !frontBuffer.isRecycled()) {
                    frontBuffer.recycle();
                }
                frontBuffer = newFrame;
            }
            if (!isUpdating) {
                isUpdating = true;
                uiHandler.post(() -> {
                    Bitmap toShow;
                    synchronized (bufferLock) {
                        toShow = frontBuffer;
                    }
                    if (toShow != null && !toShow.isRecycled() && imageView != null) {
                        imageView.setImageBitmap(toShow);
                        imageView.setVisibility(View.VISIBLE);
                        if (viewFinder != null) {
                            viewFinder.setVisibility(View.GONE);
                        }
                    }
                    isUpdating = false;
                });
            }
        }

        public void cleanup() {
            synchronized (bufferLock) {
                if (frontBuffer != null && !frontBuffer.isRecycled()) {
                    frontBuffer.recycle();
                    frontBuffer = null;
                }
            }
        }
    }
    
    // SurfaceHolder.Callback 实现
    
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
    
    // NetworkVideoCallback 实现
    
    @Override
    public void onFrame(Bitmap frame, RectF[] boxes) {
        totalFrameCount++;
        if (boxes != null) {
            totalDetectionCount += boxes.length;
        }
        // 计算FPS
        long currentTime = System.currentTimeMillis();
        if (totalFrameCount > 0) {
            currentFps = (float) totalFrameCount * 1000 / (currentTime - (currentTime - totalFrameCount * 33));
        }
        
        // 更新统计信息
        updateStatistics();

        // 显示检测结果
        synchronized (frameLock) {
            if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                lastFrameBitmap.recycle();
            }
            lastFrameBitmap = frame;
            if (uiUpdater != null && imageView != null) {
                uiUpdater.updateUI(lastFrameBitmap, imageView, surfaceNetworkPreview);
            }
        }
        Log.d(TAG, String.format("收到网络流帧: %dx%d, 检测到%d个目标", 
                frame.getWidth(), frame.getHeight(), boxes != null ? boxes.length : 0));
    }
    
    @Override
    public void onStreamStarted() {
        mainHandler.post(() -> {
            if (tvConnectionStatus != null) {
                tvConnectionStatus.setText("连接状态: 已连接");
                tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            }
            startMonitorRefresh();
            Toast.makeText(this, "网络流检测已启动", Toast.LENGTH_SHORT).show();
        });
        Log.i(TAG, "网络流检测已启动");
    }
    
    @Override
    public void onStreamStopped() {
        mainHandler.post(() -> {
            tvConnectionStatus.setText("连接状态: 已断开");
            tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            Toast.makeText(this, "网络流检测已停止", Toast.LENGTH_SHORT).show();
        });
        
        Log.i(TAG, "网络流检测已停止");
    }
    
    @Override
    public void onError(String error) {
        mainHandler.post(() -> {
            tvConnectionStatus.setText("连接状态: 错误");
            tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            Toast.makeText(this, "网络流错误: " + error, Toast.LENGTH_LONG).show();
        });
        
        Log.e(TAG, "网络流错误: " + error);
    }
    
    @Override
    public void onConnectionStatusChanged(boolean connected) {
        mainHandler.post(() -> {
            if (connected) {
                tvConnectionStatus.setText("连接状态: 已连接");
                tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            } else {
                tvConnectionStatus.setText("连接状态: 连接断开");
                tvConnectionStatus.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
            }
        });
        
        Log.i(TAG, "连接状态变化: " + (connected ? "已连接" : "已断开"));
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopMonitorRefresh();
        if (uiUpdater != null) {
            uiUpdater.cleanup();
        }
        synchronized (frameLock) {
            if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                lastFrameBitmap.recycle();
                lastFrameBitmap = null;
            }
        }
        if (networkVideoManager != null) {
            networkVideoManager.stopNetworkStream();
        }
    }
    
    @Override
    public void onBackPressed() {
        // 返回时停止检测
        stopNetworkStream();
        super.onBackPressed();
    }
}
