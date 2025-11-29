// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.LocalDetect;

import android.content.res.AssetManager;
import android.view.Surface;

public class MainAlg
{
    //加载模型
    public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu, int inputsize);
    //打开相机预览界面
    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);
    // 新增：设置置信度阈值,设置Shader拖尾开关
    public native void setThreshold(float threshold);
    public native void setNms(float nms);
    public native void setTrackEnabled(boolean enabled);
    public native void setShaderEnabled(boolean enabled);
    public native android.graphics.Bitmap detectImage(android.graphics.Bitmap bitmap, int inputSize);
    public native void setDetectEnabled(boolean enabled);
    // 新增：设置跟踪开关
    static {
        System.loadLibrary("yolov8ncnn");
    }
    public native DetectSummary getDetectSummary();
    // 防止DetectSummary被混淆裁剪的显性引用
    private static void keepDetectSummary() {
        DetectSummary dummy = new DetectSummary(0f, 0f, 0f, 0f, "");
        float t = dummy.allTimeMs + dummy.inferTimeMs + dummy.fps + dummy.cpuUsage;
        String s = dummy.logText;
        // 防止变量被优化掉
        if (t == -1f) System.out.println(s);
    }
}
