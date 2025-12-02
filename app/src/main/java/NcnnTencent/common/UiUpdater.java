package NcnnTencent.common;

import android.graphics.Bitmap;
import android.os.Handler;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ImageView;

/**
 * UI更新工具类
 * 优化UI线程的图片刷新，防止频繁刷新导致卡顿
 */
public class UiUpdater {
    private final Handler uiHandler;
    private final Object bufferLock = new Object();
    private Bitmap frontBuffer = null;
    private volatile boolean isUpdating = false;

    public UiUpdater(Handler handler) {
        this.uiHandler = handler;
    }

    /**
     * 更新UI显示
     * @param newFrame 新帧
     * @param imageView ImageView
     * @param viewFinder SurfaceView
     */
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
                    if (viewFinder != null) {
                        viewFinder.setVisibility(View.GONE);
                    }
                }
                isUpdating = false;
            });
        }
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        synchronized (bufferLock) {
            if (frontBuffer != null && !frontBuffer.isRecycled()) {
                frontBuffer.recycle();
                frontBuffer = null;
            }
        }
    }
}

