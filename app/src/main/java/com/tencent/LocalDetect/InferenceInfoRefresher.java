package com.tencent.LocalDetect;

import android.os.Handler;
import android.widget.TextView;

public class InferenceInfoRefresher {
    private final Handler handler = new Handler();
    private final TextView tvLog;
    private final MainAlg yolov8ncnn;
    private final String fixedInfo;
    private boolean running = false;

    public InferenceInfoRefresher(TextView tvLog, MainAlg yolov8ncnn, String fixedInfo) {
        this.tvLog = tvLog;
        this.yolov8ncnn = yolov8ncnn;
        this.fixedInfo = fixedInfo;
    }

    private final Runnable refresher = new Runnable() {
        @Override
        public void run() {
            // 只刷新box信息，第一行fixedInfo不变
//            String detectResult = yolov8ncnn.getLastDetectResult();
//            String log = fixedInfo + (detectResult == null ? "" : detectResult);
            tvLog.setText(fixedInfo);
            if (running) handler.postDelayed(this, 500); // 2秒刷新一次
        }
    };

    public void start() {
        if (!running) {
            running = true;
            // 先显示固定信息
            tvLog.setText(fixedInfo);
            handler.post(refresher);
        }
    }

    public void stop() {
        running = false;
        handler.removeCallbacks(refresher);
    }
}
