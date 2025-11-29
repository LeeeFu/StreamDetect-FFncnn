package com.tencent.LocalDetect;

public class DetectSummary {
    public float allTimeMs;
    public float inferTimeMs;
    public float fps;
    public float cpuUsage; // 新增字段
    public String logText;

    public DetectSummary(float allTimeMs, float inferTimeMs, float fps, float cpuUsage,  String logText) {
        this.allTimeMs = allTimeMs;
        this.inferTimeMs = inferTimeMs;
        this.fps = fps;
        this.cpuUsage = cpuUsage;
        this.logText = logText;
    }
}