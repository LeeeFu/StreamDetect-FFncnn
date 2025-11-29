project(yolov8ncnn)

cmake_minimum_required(VERSION 3.10)

# 打印ANDROID_ABI值用于调试
message(STATUS "🔍 当前ANDROID_ABI: ${ANDROID_ABI}")
message(STATUS "🔍 当前CMAKE_ANDROID_ARCH_ABI: ${CMAKE_ANDROID_ARCH_ABI}")
# 设置目标 ABI

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/eigen-3.3.9)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20231027-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/ffmpeg-5.9/arm64-v8a/include)
link_directories(${CMAKE_SOURCE_DIR}/ffmpeg-5.9/arm64-v8a/lib)
include_directories(${CMAKE_SOURCE_DIR}/x264/arm64-v8a/include)
link_directories(${CMAKE_SOURCE_DIR}/x264/arm64-v8a/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/track)
include_directories(${CMAKE_SOURCE_DIR}/detect)
include_directories(${CMAKE_SOURCE_DIR}/seg)
include_directories(${CMAKE_SOURCE_DIR}/pose)
file(GLOB TRACK_SRCS "track/*.cpp")
file(GLOB DETECT_SRCS "detect/*.cpp")
file(GLOB SEG_SRCS "seg/*.cpp")
file(GLOB POSE_SRCS "pose/*.cpp")

# 强制包含所有必要的FFmpeg静态库
set(FFMPEG_LIBS "")
set(FFMPEG_STATIC_LIBS
        libavutil.a
        libswresample.a  # 强制包含，解决swr_alloc问题
        libswscale.a
        libavcodec.a
        libavformat.a
)

# 检查x264静态库
set(X264_LIB_PATH ${CMAKE_SOURCE_DIR}/x264/arm64-v8a/lib/libx264.a)
if(EXISTS ${X264_LIB_PATH})
    file(SIZE ${X264_LIB_PATH} x264_size)
    if(x264_size GREATER 1000000)
        message(STATUS "✅ 找到x264库: libx264.a (${x264_size} bytes)")
        set(X264_LIBS ${X264_LIB_PATH})
    else()
        message(WARNING "⚠️ x264库文件太小: libx264.a (${x264_size} bytes)")
        set(X264_LIBS "")
    endif()
else()
    message(WARNING "❌ x264库不存在: ${X264_LIB_PATH}")
    set(X264_LIBS "")
endif()

# 强制检查并添加FFmpeg静态库
foreach(lib ${FFMPEG_STATIC_LIBS})
    set(lib_path ${CMAKE_SOURCE_DIR}/ffmpeg-5.9/arm64-v8a/lib/${lib})
    if(EXISTS ${lib_path})
        file(SIZE ${lib_path} lib_size)
        message(STATUS "📦 强制添加FFmpeg库: ${lib} (${lib_size} bytes)")
        list(APPEND FFMPEG_LIBS ${lib_path})
    else()
        message(ERROR "❌ 必需的FFmpeg库不存在: ${lib_path}")
        # 如果关键库不存在，禁用FFmpeg功能
        set(FFMPEG_LIBS "")
        break()
    endif()
endforeach()

# 如果静态库不可用，尝试动态库作为备选
if(NOT FFMPEG_LIBS)
    message(STATUS "🔄 静态库不可用，尝试动态库...")
    set(FFMPEG_SHARED_LIBS
            avutil
            swresample  # 强制包含音频重采样库
            swscale
            avcodec
            avformat
    )
    foreach(lib ${FFMPEG_SHARED_LIBS})
        list(APPEND FFMPEG_LIBS ${lib})
        message(STATUS "🔗 添加FFmpeg动态库: ${lib}")
    endforeach()
endif()

if(FFMPEG_LIBS)
    message(STATUS "📦 使用FFmpeg库: ${FFMPEG_LIBS}")
    if(X264_LIBS)
        message(STATUS "📦 使用x264库: ${X264_LIBS}")
    endif()
    add_library(yolov8ncnn SHARED
            main.cpp
            ffmpeg_main.cpp
            yolo_common.cpp
            ndkcamera.cpp
            globals.cpp
            ${TRACK_SRCS}
            ${DETECT_SRCS}
            ${SEG_SRCS}
            ${POSE_SRCS}
    )
else()
    message(WARNING "🚫 未找到有效的FFmpeg库，禁用FFmpeg功能")
endif()

# 链接库 - 强制包含所有必要的库
target_link_libraries(yolov8ncnn
        ncnn
        ${OpenCV_LIBS}
        camera2ndk
        mediandk
        ${FFMPEG_LIBS}
        ${X264_LIBS}
        z m c  dl )# FFmpeg依赖库