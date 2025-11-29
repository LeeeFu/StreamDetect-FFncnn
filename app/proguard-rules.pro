# 保留所有native方法声明，防止JNI相关方法被混淆
-keepclasseswithmembers class * {
    native <methods>;
}

# 保留DetectSummary类及其所有成员，防止JNI找不到构造方法
-keep class com.tencent.LocalDetect.DetectSummary { *; }

# 保留MainActivity所有内部类及其所有成员（包括匿名类）
-keep class com.tencent.LocalDetect.MainActivity$* { *; }

# 保留FFmpegDetectCallback接口及其实现
-keep interface com.tencent.LocalDetect.MainActivity$FFmpegDetectCallback { *; }