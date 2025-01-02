project(test)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/binary_predictor.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/unblur_predictor.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/unshadow_predictor.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/unwrap_predictor.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/include/opencv4"
        PUBLIC "/home/wangbo/libs/mylibs/onnxruntime-linux-x64-1.18.1/include"
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_link_libraries(${PROJECT_NAME}
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgcodecs.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_highgui.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgproc.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_core.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_videoio.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_video.so.4.8.0"
        "/home/wangbo/libs/mylibs/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so.1.18.1"
        )
