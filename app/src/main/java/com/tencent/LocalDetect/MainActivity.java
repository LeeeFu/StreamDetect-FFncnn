package com.tencent.LocalDetect;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.ImageButton;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.CheckBox;
import android.os.Process;
import android.os.SystemClock;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;

import java.io.IOException;
import java.util.Locale;
import java.io.File;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    // 帧率控制相关
    private final Object frameLock = new Object(); //互锁，，避免UI线程JNI回调线程同时操作Bitmap
    private Bitmap lastFrameBitmap = null;  //保存最近一帧做UI显示
    public static final int REQUEST_CAMERA = 100;    //打开摄像头请求码
    private SeekBar thresholdSeek, nmsSeek;   // 置信度和NMS阈值调节条
    private TextView tvThrVal, tvNmsVal;    // 显示当前阈值
    private ImageButton btnPhoto, btnVideo,  btnCamra; // 图片、视频、摄像头输入按钮
    private SurfaceView viewFinder;   // 摄像头预览窗口
    private ImageView imageView; // 用于显示图片检测结果
    private CheckBox checkboxTrack, checkboxShader;  // 跟踪和Shader开关
    private TextView tvAllValue, tvInferValue, tvFpsValue,tvCpuValue, tvLog;  //监控区和日志区
    // 状态提示相关
    private Handler statusHandler = new Handler();
    // 检测参数
    private double threshold = 0.45;
    private double nms_threshold = 0.65;
    private int facing = 1; // 默认后置摄像头
    private MainAlg yolov8ncnn = new MainAlg();// JNI接口对象
    // 新增：模型参数
    private int model,inputsize, cpugpu;      // 默认
    private String modelName,inputSize,deviceType; // "CPU" or "GPU"
    // 任务监测状态
    private volatile boolean isVideoDetecting = false;
    private boolean isImageDetecting = false;
    private boolean isCameraDetecting = false;
    private boolean isCameraInitialized = false;
    // 跟踪与Shader拖尾开关
    private boolean trackEnabled = false;
    private boolean shaderEnabled = false;

    private OptimizedUIUpdater uiUpdater;   //优化UI线程图片刷新，防止频繁刷新
    private InferenceInfoRefresher infoRefresher;  //用于刷新推理信息到监控区或者日志区

    // 新增：缓存最新的推理信息，避免频繁JNI调用
    private volatile DetectSummary cachedSummary = null;
    private final Object summaryLock = new Object();
    private volatile boolean cacheValid = false; // 新增：缓存有效性标记
    /**
     * 1、Activity生命周期入口，初始化UI、权限、模型参数、加载模型、初始化监控和日志刷新器
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        getWindow().setFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // 检查相机权限
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},103);
        }
        // 读取参数
        Intent intent = getIntent();
        model = intent.getIntExtra("model", 0);
        inputsize = intent.getIntExtra("intputsize", 0);
        cpugpu = intent.getIntExtra("cpugpu", 0);
        // 固定参数赋值
        deviceType = (cpugpu == 0) ? "CPU" : "GPU";
        String[] modelNames = getResources().getStringArray(R.array.model_list);
        if (model >= 0 && model < modelNames.length) {
            modelName = modelNames[model];
        } else {
            modelName = "UnknownModel";
        }
        String[] inputSizes = getResources().getStringArray(R.array.input_size);
        if (inputsize >= 0 && inputsize < inputSizes.length) {
            inputSize = inputSizes[inputsize];
        } else {
            inputSize = "UnknownSize";
        }
        // 加载模型
        loadYoloModel();
        initViewID();        //初始化插件
        initViewListener();  //置信度进度条联动
        // 设置初始UI状态
        imageView.setVisibility(View.GONE);
        viewFinder.setVisibility(View.VISIBLE);
        resetMonitorValues();   //设置监控区默认状态
        String fixedInfo = getLogHeader();
        infoRefresher = new InferenceInfoRefresher(tvLog, yolov8ncnn, fixedInfo);
        uiUpdater = new OptimizedUIUpdater(new Handler());
    }
    /**
     *  2、UI控件初始化,绑定所有UI控件ID
     */
    private void initViewID() {
        thresholdSeek  = findViewById(R.id.threshold_seek);
        nmsSeek = findViewById(R.id.nms_seek);
        tvThrVal = findViewById(R.id.tv_thr_val);
        tvNmsVal = findViewById(R.id.tv_nms_val);
        btnPhoto = findViewById(R.id.btn_input_image);
        btnVideo = findViewById(R.id.btn_input_video);
        btnCamra = findViewById(R.id.btn_input_camera);
        viewFinder = findViewById(R.id.view_finder);
        tvAllValue = findViewById(R.id.tv_all_value);
        tvInferValue = findViewById(R.id.tv_infer_value);
        tvFpsValue = findViewById(R.id.tv_fps_value);
        tvCpuValue = findViewById(R.id.tv_cpu_value);
        tvLog = findViewById(R.id.tv_log);
        checkboxTrack = findViewById(R.id.checkbox_track);
        checkboxShader = findViewById(R.id.checkbox_shader);
        imageView = findViewById(R.id.imageView);
        viewFinder.getHolder().addCallback(this);   //实现surfaceHolder.Callback接口，注册为SurfaceView 的回调监听
    }
    /**
     * 3、设置所有控件的事件监听器，状态量管理
     */
    private void initViewListener() {
        // 置信度进度条与数值联动
        thresholdSeek.setProgress((int) (threshold * 100));
        tvThrVal.setText(String.format(Locale.ENGLISH, "%.2f", threshold));
        thresholdSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float thresh = progress / 100.0f;
                tvThrVal.setText(String.format(Locale.ENGLISH, "%.2f", thresh));
                updateInferenceParams();      // 推理参数联动
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        // nms 监听
        nmsSeek.setProgress((int) (nms_threshold * 100));
        tvNmsVal.setText(String.format(Locale.ENGLISH, "%.2f", nms_threshold));
        nmsSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float nms = progress / 100.0f;
                tvNmsVal.setText(String.format("%.2f", nms));
                // 推理参数联动
                updateInferenceParams();
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        // 摄像头检测按钮开关逻辑
        if (btnCamra != null) {
            btnCamra.setOnClickListener(v -> {
                if (!isCameraDetecting) {
                    infoRefresher.stop();
                    clearSummaryCache();
                    // 开始检测
                    yolov8ncnn.setDetectEnabled(true);
                    startMonitorRefresh();
                    isCameraDetecting = true;
                    showStatusIndicator("已开启摄像头检测");
                } else {
                    // 停止检测，回到纯预览
                    yolov8ncnn.setDetectEnabled(false);
                    stopMonitorRefresh();
                    infoRefresher.start();
                    resetMonitorValues();
                    clearSummaryCache();
                    isCameraDetecting = false;
                    showStatusIndicator("已关闭摄像头检测");
                }
            });
        }
        if (checkboxTrack != null) {
            checkboxTrack.setOnCheckedChangeListener((buttonView, isChecked) -> {
                trackEnabled = isChecked;
                updateInferenceParams();
            });
        }
        if (checkboxShader != null) {
            checkboxShader.setOnCheckedChangeListener((buttonView, isChecked) -> {
                shaderEnabled = isChecked;
                updateInferenceParams();
            });
        }
        btnPhoto.setOnClickListener(v -> pickImage());
        btnVideo.setOnClickListener(v -> pickVideo());
    }
    /**
     * 4、选择图片，视频，从设备相册或文件管理器中选择图片的功能，包含权限检查和请求流程
     */
    private void pickImage() {
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 101);
        } else {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 101);
        }
    }
    private void pickVideo() {
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 102);
        } else {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("video/*");
            startActivityForResult(intent, 102);
        }
    }
    /**
     * 5、权限申请结果回调，权限与Activity回调，以上ActivityCompat.requestPermissions会会自动用到
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 103) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                reinitializeCamera();
                // 相机权限获取成功，可以正常使用相机
                Log.d("MainActivity", "Camera permission granted");
            } else {
                // 相机权限被拒绝
                Toast.makeText(this, "需要相机权限才能正常使用", Toast.LENGTH_LONG).show();
                Log.e("MainActivity", "Camera permission denied");
            }
        } else if (requestCode == 101 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            pickImage();
        } else if (requestCode == 102 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            pickVideo();
        }
    }
    /**
     * 6、处理图片/视频选择结果，用于接收由 startActivityForResult 启动的外部Activity（如图片选择、视频选择等）返回的结果
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) return;
        if (requestCode == 101) {
            Uri imageUri = data.getData();
            Bitmap bitmap = getPicture(imageUri);
            runImageDetect(bitmap);
        } else if (requestCode == 102) {
            Uri videoUri = data.getData();
            runVideoDetect(videoUri);
        }
    }
    /**
     * 7、SurfaceHolder.Callback创建，打开摄像头、SurfaceView尺寸变化、销毁，关闭摄像头
     */
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        yolov8ncnn.setOutputWindow(holder.getSurface());
        yolov8ncnn.openCamera(facing); // 始终有画面，保证预览不黑屏
        yolov8ncnn.setDetectEnabled(false); // 默认只预览不检测
    }
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        yolov8ncnn.setOutputWindow(holder.getSurface());
    }
    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        yolov8ncnn.closeCamera(); // 新增：释放摄像头资源
    }
    /**
     * 8、从Uri获取图片并自动旋转, 在读取图片时会用到
     */
    public Bitmap getPicture(Uri selectedImage) {
        // 1. 需要查询的列，这里只查图片的真实路径
        String[] filePathColumn = {MediaStore.Images.Media.DATA};
        // 2. 通过ContentResolver查询图片信息，返回Cursor
        Cursor cursor = this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
        if (cursor == null) return null;
        // 4. 移动到第一行（通常只有一行）
        cursor.moveToFirst();
        // 5. 获取图片路径所在的列索引,取出图片的真实文件路径, 关闭Cursor，防止内存泄漏
        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        String picturePath = cursor.getString(columnIndex);
        cursor.close();
        // 8. 用BitmapFactory解码文件为Bitmap
        Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
        if (bitmap == null) return null;
        // 10. 读取图片的旋转角度
        int rotate = readPictureDegree(picturePath);
        // 11. 按角度旋转Bitmap并返回
        return rotateBitmapByDegree(bitmap, rotate);
    }
    //读取图片文件的EXIF信息，判断图片需要旋转的角度
    public int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90: degree = 90; break;
                case ExifInterface.ORIENTATION_ROTATE_180: degree = 180; break;
                case ExifInterface.ORIENTATION_ROTATE_270: degree = 270; break;
            }
        } catch (IOException e) { e.printStackTrace(); }
        return degree;
    }
    //将Bitmap按指定角度进行旋转，返回新的Bitmap
    public Bitmap rotateBitmapByDegree(Bitmap bm, int degree) {
        Bitmap returnBm = null;
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        try {
            returnBm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) { e.printStackTrace(); }
        if (returnBm == null) returnBm = bm;
        if (bm != returnBm) bm.recycle();
        return returnBm;
    }
    /**
     * 9、加载YOLO模型,和输入尺寸解析
     */
    private void loadYoloModel() {
        boolean ret = yolov8ncnn.loadModel(getAssets(), model, cpugpu, inputsize);
        if (!ret) {
            Toast.makeText(this, "模型加载失败", Toast.LENGTH_SHORT).show();
        }
    }
    private int getInputSizeInt(String inputSize) {
        if (inputSize == null) return 320; // 默认
        String[] parts = inputSize.split("x");
        try {
            return Integer.parseInt(parts[0]);
        } catch (Exception e) {
            return 320; // 默认
        }
    }
    /**s
     * 10、检测主流程(图片、视频)
     */
    private void runImageDetect(Bitmap bitmap) {
        yolov8ncnn.closeCamera(); // 检测前关闭相机
        int size = getInputSizeInt(inputSize);
        runOnUiThread(() -> {
            imageView.setVisibility(View.VISIBLE);
            viewFinder.setVisibility(View.GONE);
            showStatusIndicator("开始图片检测");
        });
        isImageDetecting = true;
        new Thread(() -> {
            // 1. JNI层完成推理和统计
            Bitmap resultBitmap = yolov8ncnn.detectImage(bitmap, size);
            // 2. 获取JNI层统计的监控信息
            DetectSummary summary = yolov8ncnn.getDetectSummary();
            updateSummaryCache(summary); // 更新缓存
            runOnUiThread(() -> {
                imageView.setImageBitmap(resultBitmap); // 实时显示检测结果
                if (infoRefresher != null) infoRefresher.stop();
                if (summary != null) {
                    // 3. 直接显示JNI层统计的监控信息
                    tvAllValue.setText(String.format("%.0fms", summary.allTimeMs));
                    tvInferValue.setText(String.format("%.0fms", summary.inferTimeMs));
                    tvFpsValue.setText(String.format("%.1f", summary.fps));
//                    tvCpuValue.setText(String.format("%.1f%%", summary.cpuUsage));
                    tvCpuValue.setText(String.format("%.1f%%", lastCpuUsage));
                    refreshCameraLog();
                }
            });
        }).start();
    }
     // FFmpeg检测回调接口（JNI回调）
     public interface FFmpegDetectCallback {
         void onFrame(Bitmap frame, RectF[] boxes);
         void onFinish();
         void onError(String msg);
     }
     //启动FFmpeg视频检测
     public native void startFFmpegVideoDetect(String videoPath, int inputSize, FFmpegDetectCallback
         callback,android.content.res.AssetManager assetManager, int modelid, int cpugpu);

    // 视频检测主流程
    private void runVideoDetect(Uri videoUri) {
        String videoPath = getRealPathFromUri(videoUri);
        File file = new File(videoPath);
        if (!file.exists() || !file.canRead()) {
            Toast.makeText(this, "视频文件不存在或无法读取", Toast.LENGTH_LONG).show();
            return;
        }
        String lowerPath = videoPath.toLowerCase();
        if (!(lowerPath.endsWith(".mp4") || lowerPath.endsWith(".avi") || lowerPath.endsWith(".mov") || lowerPath.endsWith(".mkv"))) {
            Toast.makeText(this, "仅支持mp4/avi/mov/mkv格式", Toast.LENGTH_LONG).show();
            return;
        }
        yolov8ncnn.closeCamera(); // 检测前关闭相机
        if (isVideoDetecting) {
            Toast.makeText(this, "视频检测正在进行中，请稍候", Toast.LENGTH_SHORT).show();
            return;
        }
        int inputSizeInt = getInputSizeInt(inputSize);
        isVideoDetecting = true;
        runOnUiThread(() -> {
            imageView.setImageDrawable(null);
            imageView.setVisibility(View.VISIBLE);
            viewFinder.setVisibility(View.GONE);
            showStatusIndicator("开始视频检测");
            clearSummaryCache();
            // 恢复推理信息实时刷新
            if (infoRefresher != null && isImageDetecting) {
                infoRefresher.start();
                isImageDetecting = false;
            }
            startMonitorRefresh();            // 启动视频检测时，才开始刷新监控区
        });
        new Thread(() -> {
            try {
                startFFmpegVideoDetect(videoPath, inputSizeInt, ffmpegDetectCallback, getAssets(), model, cpugpu);
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "FFmpeg视频检测异常: " + e.getMessage(), Toast.LENGTH_LONG).show());
                isVideoDetecting = false;
                runOnUiThread(() -> {
                    showCameraPreview();
                    stopMonitorRefresh(); // 在onFinish时停止监控区定时刷新
                });
            }
        }).start();
    }
    //FFmpeg检测回调实例
    private final FFmpegDetectCallback ffmpegDetectCallback = new FFmpegDetectCallback() {
        @Override
        public void onFrame(Bitmap frame, RectF[] boxes) {
            // 更新推理信息缓存（JNI层每帧都会更新summary）
            DetectSummary summary = yolov8ncnn.getDetectSummary();
            if (summary != null) {
                updateSummaryCache(summary);
            }
            // 只更新图像，不刷新日志区
            synchronized (frameLock) {
                // 回收上一个Bitmap
                if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                    lastFrameBitmap.recycle();
                }
                lastFrameBitmap = frame;
                // 确保每帧都显示imageView，隐藏viewFinder
                runOnUiThread(() -> {
                    imageView.setVisibility(View.VISIBLE);
                    viewFinder.setVisibility(View.GONE);
                    imageView.setImageBitmap(frame);
                });
                uiUpdater.updateUI(lastFrameBitmap, imageView, viewFinder);
            }
        }
        @Override
        public void onFinish() {
            runOnUiThread(() -> {
                // 恢复UI状态
                imageView.setVisibility(View.GONE);
                viewFinder.setVisibility(View.VISIBLE);
                // 清理资源
                synchronized (frameLock) {
                    if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                        lastFrameBitmap.recycle();
                        lastFrameBitmap = null;
                    }
                }
                isVideoDetecting = false;
                showStatusIndicator("视频检测完成");
                stopMonitorRefresh(); // 结束时停止定时器
            });
        }
        @Override
        public void onError(String msg) {
                isVideoDetecting = false;
                runOnUiThread(() -> {
                    showStatusIndicator("视频检测失败: " + msg);
                    showCameraPreview();
                    stopMonitorRefresh();
            });
        }
    };
    /**
     * 11、工具函数， Uri转文件路径、切换回摄像头预览、实时推理参数联动方法
     */
    private String getRealPathFromUri(Uri uri) {
        String[] proj = { MediaStore.Images.Media.DATA };
        Cursor cursor = getContentResolver().query(uri, proj, null, null, null);
        if (cursor != null) {
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            String path = cursor.getString(column_index);
            cursor.close();
            return path;
        }
        return uri.getPath();
    }
    private void showCameraPreview() {
        runOnUiThread(() -> {
            imageView.setVisibility(View.GONE);
            viewFinder.setVisibility(View.VISIBLE);
            yolov8ncnn.closeCamera(); // 关闭摄像头，彻底关闭推理
            resetMonitorValues();
            stopMonitorRefresh();
            showStatusIndicator("切换到摄像头预览");
        });
    }
    //
    private void updateInferenceParams() {
        float thresh = thresholdSeek.getProgress() / 100.0f;
        float nms = nmsSeek.getProgress() / 100.0f;
        yolov8ncnn.setThreshold((float) thresh);
        yolov8ncnn.setNms((float) nms);
        yolov8ncnn.setTrackEnabled(trackEnabled);
        yolov8ncnn.setShaderEnabled(shaderEnabled);
    }
    /**
     * 状态提示相关方法
     */
    private void showStatusIndicator(String message) {
        runOnUiThread(() -> {
//            Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
            Toast toast = Toast.makeText(this, message, Toast.LENGTH_SHORT);
            toast.show();
            // 1秒后自动取消Toast
            new Handler().postDelayed(() -> {
                try {
                    toast.cancel();
                } catch (Exception e) {
                    // 忽略异常
                }
            }, 1000); // 可以修改这个数值来控制显示时间（毫秒）
        });
    }
    /**
     * 12、工监控区刷新相关
     */
    // 1. 新增监控区定时刷新逻辑
    private final Handler monitorHandler = new Handler();
    private long lastCpuTime = 0, lastRealTime = 0;
    private float lastCpuUsage = 0f;
    private static final String TAG_CPU = "CPU_MONITOR";

    private final Runnable monitorRefresher = new Runnable() {
        @Override
        public void run() {
            long cpuTime = Process.getElapsedCpuTime();
            long realTime = SystemClock.elapsedRealtime();
            if (lastRealTime > 0) {
                long cpuDelta = cpuTime - lastCpuTime;
                long realDelta = realTime - lastRealTime;
                if (realDelta > 0) {
                    // 进程核数百分比：满载1核=100%，满载8核=800%
                    lastCpuUsage = 100f * cpuDelta / (realDelta * 10);
                    Log.d(TAG_CPU, String.format("cpuDelta=%d, realDelta=%d, cpuUsage=%.1f%% (核数百分比)", cpuDelta, realDelta, lastCpuUsage));
                }
            }
            lastCpuTime = cpuTime;
            lastRealTime = realTime;
            // 直接读取缓存中的最新推理信息，避免频繁JNI调用
            DetectSummary summary = getLatestSummary();
            if (summary != null) {
                tvAllValue.setText(String.format("%.0fms", summary.allTimeMs));
                tvInferValue.setText(String.format("%.0fms", summary.inferTimeMs));
                tvFpsValue.setText(String.format("%.1f", summary.fps));
//                tvCpuValue.setText(String.format("%.1f%%", summary.cpuUsage));
                tvCpuValue.setText(String.format("%.1f%%", lastCpuUsage)); // 用核数百分比
                refreshCameraLog();// <--- 新增，显示类别统计日志
            }
            monitorHandler.postDelayed(this, 1000); // 1秒刷新一次
        }
    };
    // 获取最新推理信息的缓存方法
    private DetectSummary getLatestSummary() {
        synchronized (summaryLock) {
            // 在摄像头检测或视频检测时，都实时获取最新的summary
            if (isCameraDetecting || isVideoDetecting) {
                DetectSummary latest = yolov8ncnn.getDetectSummary();
                if (latest != null) {
                    cachedSummary = latest;
                    cacheValid = true;
                }
            } else if (cachedSummary == null || !cacheValid) {
                cachedSummary = yolov8ncnn.getDetectSummary();
                cacheValid = (cachedSummary != null);
            }
            return cachedSummary;
        }
    }

    // 新增：清理缓存的方法
    private void clearSummaryCache() {
        synchronized (summaryLock) {
            cachedSummary = null;
            cacheValid = false;
        }
    }
    // 更新推理信息缓存的方法（可由JNI回调或推理完成后调用）
    public void updateSummaryCache(DetectSummary newSummary) {
        synchronized (summaryLock) {
            cachedSummary = newSummary;
            cacheValid = true;
        }
    }
    //启动监控区定时刷新, // 在摄像头/视频流推理开始时启动监控区定时刷新
    private void startMonitorRefresh() {
        monitorHandler.removeCallbacks(monitorRefresher);
        lastCpuTime = Process.getElapsedCpuTime();
        lastRealTime = SystemClock.elapsedRealtime();
        lastCpuUsage = 0f; // 重置CPU使用率
        monitorHandler.post(monitorRefresher);
    }
    //停止监控区定时刷新
    private void stopMonitorRefresh() {
        monitorHandler.removeCallbacks(monitorRefresher);
    }
    //重置监控区显示
    private void resetMonitorValues() {
        tvAllValue.setText("--");
        tvInferValue.setText("--");
        tvFpsValue.setText("--");
        tvCpuValue.setText("--");
        // 重置CPU统计变量
        lastCpuUsage = 0f;
        lastCpuTime = 0;
        lastRealTime = 0;
    }
    // 日志区头部信息
    private String getLogHeader() {
        String device = deviceType != null ? deviceType : "CPU";
        String model = modelName != null ? modelName : "Model";
        String size = inputSize != null ? inputSize : "";
        return String.format("[%s] %s, input_size: %s", device, model, size);
    }
    // 优化后的摄像头检测日志区刷新逻辑
    private void refreshCameraLog() {
        DetectSummary summary = getLatestSummary(); // 使用缓存方法
        String logHeader = getLogHeader();
        String logBody = summary != null ? summary.logText : "";
        if (summary != null) {
            if (logBody != null && !logBody.trim().isEmpty()) {
                tvLog.setText(logHeader + "\n" + logBody);
            } else {
                tvLog.setText(logHeader + "\nDetect_Info: 0 target");
            }
        }
    }
    /**
     * 13、线程安全的UI刷新机制
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
                // 回收旧的
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
                    if (toShow != null && !toShow.isRecycled()) {
                        imageView.setImageBitmap(toShow);
                        imageView.setVisibility(View.VISIBLE);
                        viewFinder.setVisibility(View.GONE);
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
    /**
     * 14、线程安全的UI刷新机制
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (infoRefresher != null) infoRefresher.stop();
        uiUpdater.cleanup();
        // 清理视频检测相关资源
        synchronized (frameLock) {
            if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                lastFrameBitmap.recycle();
                lastFrameBitmap = null;
            }
        }
        clearSummaryCache();
        // 停止视频检测
        isVideoDetecting = false;
        stopMonitorRefresh(); // 在onDestroy时停止监控区定时刷新
    }
    /**
     * 初始化摄像头
     */
    private void initializeCamera(SurfaceHolder holder) {
        try {
            yolov8ncnn.setOutputWindow(holder.getSurface());
            yolov8ncnn.openCamera(facing); // 始终有画面，保证预览不黑屏
            yolov8ncnn.setDetectEnabled(false); // 默认只预览不检测
            isCameraInitialized = true; // 设置初始化成功标志
            Log.d("MainActivity", "Camera initialized successfully");
        } catch (Exception e) {
            isCameraInitialized = false;
            Log.e("MainActivity", "Failed to initialize camera: " + e.getMessage());
        }
    }
    /**
     * 重新初始化摄像头（权限获取后调用）
     */
    private void reinitializeCamera() {
        if (viewFinder != null && viewFinder.getHolder() != null) {
            SurfaceHolder holder = viewFinder.getHolder();
            if (holder.getSurface().isValid()) {
                initializeCamera(holder);
            } else {
                Log.d("MainActivity", "Surface not ready, will initialize when surface is created");
                // 延迟重新初始化，确保Surface已经准备好
                new Handler().postDelayed(() -> {
                    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        reinitializeCamera();
                    }
                }, 500); // 延迟500ms
            }
        } else {
            Log.d("MainActivity", "ViewFinder or Holder not ready, will retry later");
            // 如果Surface还没有准备好，延迟重试
            new Handler().postDelayed(() -> {
                if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    reinitializeCamera();
                }
            }, 1000); // 延迟1秒
        }
    }
}
