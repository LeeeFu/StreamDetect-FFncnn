package NcnnTencent;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import NcnnTencent.LocalDetect.R;

/**
 * 欢迎界面主函数
 * 负责模型参数选择和启动本地/云端检测
 */
public class WelcomeActivity extends Activity {
    private Spinner spinnerModel;                    //模型选择下拉框
    private Spinner spinnerInputsize;                //输入尺寸选择下拉框
    //默认参数
    private static final int DEFAULT_MODEL = 0;      //默认模型索引（0=high_speed）
    private static final int DEFAULT_INPUTSIZE = 0;  // 默认输入尺寸索引（0=320x320）
    // 参数选择状态
    private boolean modelChanged = false;            //跟踪模型是否被修改
    private boolean inputsizeChanged = false;        //跟踪输入尺寸是否被修改
    private boolean isCPU = true;                   // 默认CPU，当前设备类型
    private ImageView imgDevice;                    //CPU/GPU切换图标

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome);     //设置布局文件
        initViews();   //初始化相关组件
        setupDeviceToggle();  //cpu/gpu切换
        setupDefaultSelections(); //模型默认参数选择，后续可更改成记忆功能
        setupSpinnerListeners();  //追踪模型尺寸等信息的选择
        setupButtonListeners();  // 进入主程序
    }
    /** 初始化UI组件 **/
    private void initViews() {
        spinnerModel = findViewById(R.id.btn_model_manage);
        spinnerInputsize = findViewById(R.id.btn_size_manage);
        imgDevice = findViewById(R.id.img_device);
    }
    /** 设置设备切换功能 **/
    private void setupDeviceToggle() {
        imgDevice.setOnClickListener(v -> {
            isCPU = !isCPU;
            imgDevice.setImageResource(isCPU ? R.drawable.ic_cpu : R.drawable.ic_gpu);
            Toast.makeText(this, isCPU ? "已切换为CPU推理" : "已切换为GPU推理", Toast.LENGTH_SHORT).show();
        });
    }
    /*** 设置默认选择 */
    private void setupDefaultSelections() {
        spinnerModel.setSelection(DEFAULT_MODEL);
        spinnerInputsize.setSelection(DEFAULT_INPUTSIZE);
    }
    /** 设置下拉框监听器 */
    private void setupSpinnerListeners() {
        // 模型选择监听器
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position != DEFAULT_MODEL) {
                    modelChanged = true;
                    Log.d("WelcomeActivity", "Model changed to: " + position);
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 无需处理
            }
        });

        // 输入尺寸选择监听器
        spinnerInputsize.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position != DEFAULT_INPUTSIZE) {
                    inputsizeChanged = true;
                    Log.d("WelcomeActivity", "Input size changed to: " + position);
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 无需处理
            }
        });
    }
    /** * 设置按钮监听器  */
    private void setupButtonListeners() {
        Button btnDetect = findViewById(R.id.btn_detect);
        btnDetect.setOnClickListener(v -> startLocalDetection());

        findViewById(R.id.btn_segment).setOnClickListener(v -> startCloudDetection());
    }
    /** * 启动云端检测功能 */
    private void startCloudDetection() {
        // 获取当前选择的参数
        int model = spinnerModel.getSelectedItemPosition();
        int inputsize = spinnerInputsize.getSelectedItemPosition();

        // 检查是否有任何参数被修改
        boolean hasCustomSettings = modelChanged || inputsizeChanged;
        // 参数验证和默认值设置
        if (!hasCustomSettings) {
            // 如果没有修改参数，使用默认值
            model = DEFAULT_MODEL;
            inputsize = DEFAULT_INPUTSIZE;
            Log.d("WelcomeActivity", "Using default parameters for cloud: model=" + model + ", inputsize=" + inputsize);
        }

        // 启动MainCloudActivity
        Intent intent = new Intent(this, NcnnTencent.CloudDetect.MainCloudActivity.class);
        intent.putExtra("model", model);
        intent.putExtra("inputsize", inputsize);
        int cpugpu = isCPU ? 0 : 1; // 0=CPU, 1=GPU
        intent.putExtra("cpugpu", cpugpu);
        startActivity(intent);
    }
    /**  * 启动本地检测功能 */
    private void startLocalDetection() {
        // 获取当前选择的参数
        int model = spinnerModel.getSelectedItemPosition();
        int inputsize = spinnerInputsize.getSelectedItemPosition();

        // 检查是否有任何参数被修改
        boolean hasCustomSettings = modelChanged || inputsizeChanged;
        // 参数验证和默认值设置
        if (!hasCustomSettings) {
            // 如果没有修改参数，使用默认值
            model = DEFAULT_MODEL;
            inputsize = DEFAULT_INPUTSIZE;
            Log.d("WelcomeActivity", "Using default parameters: model=" + model + ", inputsize=" + inputsize);
        }
        // 显示参数信息
        showParameterInfo(model, inputsize);
        // 启动MainLocalActivity
        startMainLocalActivity(model, inputsize, hasCustomSettings);
    }
    /*** 显示参数信息*/
    private void showParameterInfo(int model, int inputsize) {
        String[] modelNames = getResources().getStringArray(R.array.model_list);
        String[] sizeNames = getResources().getStringArray(R.array.input_size);

        String paramInfo = String.format("参数设置: 模型-%s, %s",
                modelNames[model], sizeNames[inputsize]);
        Toast.makeText(this, paramInfo, Toast.LENGTH_SHORT).show();
    }
    /** * 启动MainLocalActivity */
    private void startMainLocalActivity(int model, int inputsize, boolean hasCustomSettings) {
        Intent intent = new Intent(this, NcnnTencent.LocalDetect.MainLocalActivity.class);
        intent.putExtra("model", model);
        intent.putExtra("intputsize", inputsize);
        int cpugpu = isCPU ? 0 : 1; // 0=CPU, 1=GPU
        intent.putExtra("cpugpu", cpugpu);
        intent.putExtra("hasCustomSettings", hasCustomSettings);
        startActivity(intent);
    }
}

