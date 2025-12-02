package NcnnTencent.common;

import android.os.Handler;
import android.os.Process;
import android.os.SystemClock;
import android.widget.TextView;

/**
 * 监控器类集合
 * 整合推理监控和系统监控功能，简化项目结构
 */
public class Monitor {

    /**
     * 系统监控工具类
     * 用于监控CPU使用率等系统指标
     */
    public static class SystemMonitor {
        private Handler handler;
        private long lastCpuTime = 0;
        private long lastRealTime = 0;
        private float lastCpuUsage = 0f;
        private Runnable monitorTask;
        private boolean isMonitoring = false;

        public interface MonitorCallback {
            void onCpuUsageUpdated(float cpuUsage);
        }

        private MonitorCallback callback;

        public SystemMonitor(Handler handler) {
            this.handler = handler;
        }

        public void setCallback(MonitorCallback callback) {
            this.callback = callback;
        }

        /**
         * 开始监控
         */
        public void start() {
            if (isMonitoring) return;
            isMonitoring = true;
            lastCpuTime = Process.getElapsedCpuTime();
            lastRealTime = SystemClock.elapsedRealtime();
            lastCpuUsage = 0f;

            monitorTask = new Runnable() {
                @Override
                public void run() {
                    long cpuTime = Process.getElapsedCpuTime();
                    long realTime = SystemClock.elapsedRealtime();
                    if (lastRealTime > 0) {
                        long cpuDelta = cpuTime - lastCpuTime;
                        long realDelta = realTime - lastRealTime;
                        if (realDelta > 0) {
                            int cpuCores = Runtime.getRuntime().availableProcessors();
                            lastCpuUsage = 100f * cpuDelta / (realDelta * cpuCores);
                        }
                    }
                    lastCpuTime = cpuTime;
                    lastRealTime = realTime;

                    if (callback != null) {
                        callback.onCpuUsageUpdated(lastCpuUsage);
                    }

                    if (isMonitoring) {
                        handler.postDelayed(this, 1000); // 1秒刷新一次
                    }
                }
            };
            handler.post(monitorTask);
        }

        /**
         * 停止监控
         */
        public void stop() {
            isMonitoring = false;
            if (monitorTask != null) {
                handler.removeCallbacks(monitorTask);
            }
        }

        /**
         * 获取当前CPU使用率
         * @return CPU使用率百分比
         */
        public float getCurrentCpuUsage() {
            return lastCpuUsage;
        }

        /**
         * 重置统计
         */
        public void reset() {
            lastCpuUsage = 0f;
            lastCpuTime = 0;
            lastRealTime = 0;
        }
    }

    /**
     * 推理监控器
     * 统一管理推理信息的刷新和显示
     */
    public static class InferenceMonitor {
        private Handler handler;
        private TextView tvAllValue;
        private TextView tvInferValue;
        private TextView tvFpsValue;
        private TextView tvCpuValue;
        private TextView tvLog;
        private JniBridge jniBridge;
        private SystemMonitor systemMonitor;
        private String logHeader;
        private Runnable refreshTask;
        private boolean isMonitoring = false;

        public InferenceMonitor(Handler handler, JniBridge jniBridge) {
            this.handler = handler;
            this.jniBridge = jniBridge;
            this.systemMonitor = new SystemMonitor(handler);
            this.systemMonitor.setCallback(cpuUsage -> {
                if (tvCpuValue != null) {
                    tvCpuValue.setText(String.format("%.1f%%", cpuUsage));
                }
            });
        }

        /**
         * 设置UI控件
         */
        public void setViews(TextView tvAllValue, TextView tvInferValue,
                             TextView tvFpsValue, TextView tvCpuValue, TextView tvLog) {
            this.tvAllValue = tvAllValue;
            this.tvInferValue = tvInferValue;
            this.tvFpsValue = tvFpsValue;
            this.tvCpuValue = tvCpuValue;
            this.tvLog = tvLog;
        }

        /**
         * 设置日志头部信息
         */
        public void setLogHeader(String header) {
            this.logHeader = header;
        }

        /**
         * 开始监控
         */
        public void start() {
            if (isMonitoring) return;
            isMonitoring = true;
            systemMonitor.start();

            refreshTask = new Runnable() {
                @Override
                public void run() {
                    Models.DetectSummary summary = jniBridge.getDetectSummary();
                    if (summary != null) {
                        updateViews(summary);
                    }
                    if (isMonitoring) {
                        handler.postDelayed(this, 1000); // 1秒刷新一次
                    }
                }
            };
            handler.post(refreshTask);
        }

        /**
         * 停止监控
         */
        public void stop() {
            isMonitoring = false;
            systemMonitor.stop();
            if (refreshTask != null) {
                handler.removeCallbacks(refreshTask);
            }
        }

        /**
         * 更新视图（必须在UI线程中调用）
         */
        private void updateViews(Models.DetectSummary summary) {
            // refreshTask已经在handler线程中运行，所以这里直接更新UI，不需要再post
            if (tvAllValue != null) {
                tvAllValue.setText(String.format("%.0fms", summary.allTimeMs));
            }
            if (tvInferValue != null) {
                tvInferValue.setText(String.format("%.0fms", summary.inferTimeMs));
            }
            if (tvFpsValue != null) {
                tvFpsValue.setText(String.format("%.1f", summary.fps));
            }
            if (tvLog != null) {
                String logBody = summary.logText != null ? summary.logText : "";
                if (logBody.trim().isEmpty()) {
                    logBody = "Detect_Info: 0 target";
                }
                tvLog.setText(logHeader + "\n" + logBody);
            }
        }

        /**
         * 重置显示
         */
        public void reset() {
            if (tvAllValue != null) tvAllValue.setText("--");
            if (tvInferValue != null) tvInferValue.setText("--");
            if (tvFpsValue != null) tvFpsValue.setText("--");
            if (tvCpuValue != null) tvCpuValue.setText("--");
            if (tvLog != null && logHeader != null) {
                tvLog.setText(logHeader);
            }
            systemMonitor.reset();
        }

        /**
         * 更新单次检测结果（用于图片检测）
         */
        public void updateSingleResult(Models.DetectSummary summary) {
            // 确保在UI线程中更新
            handler.post(() -> {
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
                    tvCpuValue.setText(String.format("%.1f%%", systemMonitor.getCurrentCpuUsage()));
                }
                if (tvLog != null) {
                    String logBody = summary.logText != null ? summary.logText : "";
                    if (logBody.trim().isEmpty()) {
                        logBody = "Detect_Info: 0 target";
                    }
                    tvLog.setText(logHeader + "\n" + logBody);
                }
            });
        }
    }
}

