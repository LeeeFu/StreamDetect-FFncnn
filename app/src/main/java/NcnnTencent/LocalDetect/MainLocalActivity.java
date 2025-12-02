package NcnnTencent.LocalDetect;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;

import NcnnTencent.common.JniBridge;
import NcnnTencent.common.Monitor.InferenceMonitor;
import NcnnTencent.common.Models.DetectSummary;
import NcnnTencent.common.UiUpdater;
import NcnnTencent.LocalDetect.presenter.MainPresenter;

/**
 * MainLocalActivity - 本地检测主函数
 * 采用MVP架构，只负责UI展示和用户交互，业务逻辑由Presenter处理
 */
public class MainLocalActivity extends Activity implements
        SurfaceHolder.Callback,
        MainPresenter.View,
        MainPresenter.VideoDetectCallback {

    // UI控件
    private SeekBar thresholdSeek, nmsSeek;
    private TextView tvThrVal, tvNmsVal;
    private ImageButton btnPhoto, btnVideo, btnCamera;
    private SurfaceView viewFinder;
    private ImageView imageView;
    private CheckBox checkboxTrack, checkboxShader;
    private TextView tvAllValue, tvInferValue, tvFpsValue, tvCpuValue, tvLog;

    // 架构组件
    private JniBridge jniBridge;
    private MainPresenter presenter;
    private InferenceMonitor inferenceMonitor;
    private UiUpdater uiUpdater;
    private Handler mainHandler;

    // 状态管理
    private final Object frameLock = new Object();
    private Bitmap lastFrameBitmap = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        getWindow().setFlags(
                android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        );

        // 检查相机权限
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 103);
        }

        // 初始化架构组件
        mainHandler = new Handler();
        jniBridge = new JniBridge();
        presenter = new MainPresenter(this, this, jniBridge);
        inferenceMonitor = new InferenceMonitor(mainHandler, jniBridge);
        uiUpdater = new UiUpdater(mainHandler);

        // 读取参数并初始化
        Intent intent = getIntent();
        int model = intent.getIntExtra("model", 0);
        int inputsize = intent.getIntExtra("intputsize", 0);
        int cpugpu = intent.getIntExtra("cpugpu", 0);
        presenter.initializeConfig(model, inputsize, cpugpu);

        // 初始化UI
        initViews();
        setupListeners();
        setupMonitor();

        // 设置初始UI状态
        imageView.setVisibility(View.GONE);
        viewFinder.setVisibility(View.VISIBLE);
        inferenceMonitor.reset();
    }

    /**
     * 初始化UI控件
     */
    private void initViews() {
        thresholdSeek = findViewById(R.id.threshold_seek);
        nmsSeek = findViewById(R.id.nms_seek);
        tvThrVal = findViewById(R.id.tv_thr_val);
        tvNmsVal = findViewById(R.id.tv_nms_val);
        btnPhoto = findViewById(R.id.btn_input_image);
        btnVideo = findViewById(R.id.btn_input_video);
        btnCamera = findViewById(R.id.btn_input_camera);
        viewFinder = findViewById(R.id.view_finder);
        tvAllValue = findViewById(R.id.tv_all_value);
        tvInferValue = findViewById(R.id.tv_infer_value);
        tvFpsValue = findViewById(R.id.tv_fps_value);
        tvCpuValue = findViewById(R.id.tv_cpu_value);
        tvLog = findViewById(R.id.tv_log);
        checkboxTrack = findViewById(R.id.checkbox_track);
        checkboxShader = findViewById(R.id.checkbox_shader);
        imageView = findViewById(R.id.imageView);
        viewFinder.getHolder().addCallback(this);
    }

    /**
     * 设置监听器
     */
    private void setupListeners() {
        // 置信度进度条
        thresholdSeek.setProgress(45);
        tvThrVal.setText("0.45");
        thresholdSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float thresh = progress / 100.0f;
                tvThrVal.setText(String.format("%.2f", thresh));
                updateInferenceParams();
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // NMS进度条
        nmsSeek.setProgress(65);
        tvNmsVal.setText("0.65");
        nmsSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float nms = progress / 100.0f;
                tvNmsVal.setText(String.format("%.2f", nms));
                updateInferenceParams();
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // 按钮监听
        btnCamera.setOnClickListener(v -> presenter.toggleCameraDetection());
        btnPhoto.setOnClickListener(v -> pickImage());
        btnVideo.setOnClickListener(v -> pickVideo());

        checkboxTrack.setOnCheckedChangeListener((buttonView, isChecked) -> updateInferenceParams());
        checkboxShader.setOnCheckedChangeListener((buttonView, isChecked) -> updateInferenceParams());
    }

    /**
     * 设置监控器
     */
    private void setupMonitor() {
        inferenceMonitor.setViews(tvAllValue, tvInferValue, tvFpsValue, tvCpuValue, tvLog);
        String logHeader = getLogHeader();
        inferenceMonitor.setLogHeader(logHeader);
    }

    /**
     * 更新推理参数
     */
    private void updateInferenceParams() {
        float threshold = thresholdSeek.getProgress() / 100.0f;
        float nms = nmsSeek.getProgress() / 100.0f;
        boolean trackEnabled = checkboxTrack.isChecked();
        boolean shaderEnabled = checkboxShader.isChecked();
        presenter.updateInferenceParams(threshold, nms, trackEnabled, shaderEnabled);
    }

    /**
     * 选择图片
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

    /**
     * 选择视频
     */
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
     * 权限申请结果回调
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 103) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                reinitializeCamera();
            } else {
                Toast.makeText(this, "需要相机权限才能正常使用", Toast.LENGTH_LONG).show();
            }
        } else if (requestCode == 101 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            pickImage();
        } else if (requestCode == 102 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            pickVideo();
        }
    }

    /**
     * Activity结果回调
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) return;
        if (requestCode == 101) {
            Uri imageUri = data.getData();
            presenter.handleImageDetection(imageUri);
        } else if (requestCode == 102) {
            Uri videoUri = data.getData();
            presenter.handleVideoDetection(videoUri, this);
        }
    }

    /**
     * SurfaceHolder.Callback实现
     */
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        presenter.onSurfaceCreated(holder.getSurface());
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        presenter.onSurfaceChanged(holder.getSurface());
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        presenter.onSurfaceDestroyed();
    }

    /**
     * MainPresenter.View接口实现
     */
    @Override
    public void showImageResult(Bitmap bitmap) {
        runOnUiThread(() -> {
            synchronized (frameLock) {
                if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                    lastFrameBitmap.recycle();
                }
                lastFrameBitmap = bitmap;
            }
            uiUpdater.updateUI(bitmap, imageView, viewFinder);
        });
    }

    @Override
    public void showStatusMessage(String message) {
        runOnUiThread(() -> {
            Toast toast = Toast.makeText(this, message, Toast.LENGTH_SHORT);
            toast.show();
            new Handler().postDelayed(() -> {
                try {
                    toast.cancel();
                } catch (Exception e) {
                    // 忽略异常
                }
            }, 1000);
        });
    }

    @Override
    public void showError(String error) {
        runOnUiThread(() -> Toast.makeText(this, error, Toast.LENGTH_LONG).show());
    }

    @Override
    public void updateMonitor(DetectSummary summary) {
        runOnUiThread(() -> inferenceMonitor.updateSingleResult(summary));
    }

    @Override
    public void switchToImageView() {
        runOnUiThread(() -> {
            imageView.setVisibility(View.VISIBLE);
            viewFinder.setVisibility(View.GONE);
        });
    }

    @Override
    public void switchToCameraView() {
        runOnUiThread(() -> {
            imageView.setVisibility(View.GONE);
            viewFinder.setVisibility(View.VISIBLE);
        });
    }

    @Override
    public void startMonitoring() {
        runOnUiThread(() -> inferenceMonitor.start());
    }

    @Override
    public void stopMonitoring() {
        runOnUiThread(() -> {
            inferenceMonitor.stop();
            inferenceMonitor.reset();
        });
    }

    /**
     * MainPresenter.VideoDetectCallback接口实现
     */
    @Override
    public void startVideoDetect(String videoPath, int inputSize,
                                 android.content.res.AssetManager assetManager,
                                 int modelId, int deviceType) {
        new Thread(() -> {
            try {
                jniBridge.startFFmpegVideoDetect(videoPath, inputSize,
                        new JniBridge.FFmpegDetectCallback() {
                            @Override
                            public void onFrame(Bitmap frame, RectF[] boxes) {
                                presenter.onVideoFrameReceived(frame, boxes);
                            }
                            @Override
                            public void onFinish() {
                                presenter.onVideoDetectFinish();
                            }
                            @Override
                            public void onError(String msg) {
                                presenter.onVideoDetectError(msg);
                            }
                        }, assetManager, modelId, deviceType);
            } catch (Exception e) {
                presenter.onVideoDetectError("FFmpeg视频检测异常: " + e.getMessage());
            }
        }).start();
    }

    /**
     * 重新初始化摄像头
     */
    private void reinitializeCamera() {
        if (viewFinder != null && viewFinder.getHolder() != null) {
            SurfaceHolder holder = viewFinder.getHolder();
            if (holder.getSurface().isValid()) {
                presenter.onSurfaceCreated(holder.getSurface());
            } else {
                new Handler().postDelayed(() -> {
                    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                            == PackageManager.PERMISSION_GRANTED) {
                        reinitializeCamera();
                    }
                }, 500);
            }
        }
    }

    /**
     * 获取日志头部信息
     */
    private String getLogHeader() {
        Intent intent = getIntent();
        int model = intent.getIntExtra("model", 0);
        int inputsize = intent.getIntExtra("intputsize", 0);
        int cpugpu = intent.getIntExtra("cpugpu", 0);

        String device = (cpugpu == 0) ? "CPU" : "GPU";
        String[] modelNames = getResources().getStringArray(R.array.model_list);
        String[] inputSizes = getResources().getStringArray(R.array.input_size);

        String modelName = (model >= 0 && model < modelNames.length) ? modelNames[model] : "UnknownModel";
        String inputSize = (inputsize >= 0 && inputsize < inputSizes.length) ? inputSizes[inputsize] : "";

        return String.format("[%s] %s, input_size: %s", device, modelName, inputSize);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        presenter.cleanup();
        inferenceMonitor.stop();
        uiUpdater.cleanup();
        synchronized (frameLock) {
            if (lastFrameBitmap != null && !lastFrameBitmap.isRecycled()) {
                lastFrameBitmap.recycle();
                lastFrameBitmap = null;
            }
        }
    }
}

